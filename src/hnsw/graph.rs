//! HNSW graph — core data structures and algorithms.
//!
//! Implements the Hierarchical Navigable Small World graph from:
//! "Efficient and robust approximate nearest neighbor search using
//!  Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2016/2018).

use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::distance::DistanceMetric;
use crate::error::{Result, VectorDbError};
use crate::vector::Vector;

use super::neighbor_queue::{MaxHeap, MinHeap, Neighbor};

/// Configuration parameters for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Max number of connections per node (layers > 0).
    pub m: usize,
    /// Max connections at layer 0 (typically 2 * m).
    pub m_max0: usize,
    /// Number of candidates during construction.
    pub ef_construction: usize,
    /// Number of candidates during search.
    pub ef_search: usize,
    /// Level generation factor: 1 / ln(m).
    pub ml: f64,
    /// Maximum number of layers.
    pub max_layers: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max0: 2 * m,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
            max_layers: 16,
        }
    }
}

impl HnswParams {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            max_layers: 16,
        }
    }
}

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
struct HnswNode {
    #[allow(dead_code)]
    id: usize,
    vector: Vector,
    /// Neighbors per layer. neighbors[l] is the list of neighbor IDs at layer l.
    neighbors: Vec<Vec<usize>>,
    /// The maximum layer this node was inserted into.
    level: usize,
}

/// The HNSW graph structure.
#[derive(Debug)]
pub struct HnswGraph {
    /// Nodes indexed by internal ID. Slots can be None after deletion.
    nodes: Vec<Option<HnswNode>>,
    /// Entry point node ID (highest-level node).
    entry_point: Option<usize>,
    /// Current maximum level in the graph.
    max_level: usize,
    /// HNSW parameters.
    params: HnswParams,
    /// Distance metric.
    metric: DistanceMetric,
    /// RNG for level generation.
    rng: StdRng,
    /// Count of active (non-deleted) nodes.
    count: usize,
}

impl HnswGraph {
    pub fn new(metric: DistanceMetric, params: HnswParams) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            params,
            metric,
            rng: StdRng::from_entropy(),
            count: 0,
        }
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Generate a random level for a new node.
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let level = (-r.ln() * self.params.ml).floor() as usize;
        level.min(self.params.max_layers - 1)
    }

    /// Compute distance between a query vector and a node.
    fn distance(&self, query: &Vector, node_id: usize) -> Result<f32> {
        let node = self.nodes[node_id]
            .as_ref()
            .ok_or_else(|| VectorDbError::IndexError("Node not found".to_string()))?;
        self.metric.distance(query, &node.vector)
    }

    /// Get the vector for a given node ID (for internal use).
    pub fn get_vector(&self, id: usize) -> Option<&Vector> {
        self.nodes.get(id).and_then(|n| n.as_ref()).map(|n| &n.vector)
    }

    /// SEARCH-LAYER: Algorithm 2 from the HNSW paper.
    ///
    /// Search a single layer of the graph for the ef closest neighbors to query.
    /// `ep` is the set of entry points (their IDs).
    /// Returns the ef closest neighbors found.
    fn search_layer(
        &self,
        query: &Vector,
        ep: &[usize],
        ef: usize,
        layer: usize,
    ) -> Result<Vec<Neighbor>> {
        let mut visited = HashSet::new();
        let mut candidates = MinHeap::new(); // closest candidate on top
        let mut results = MaxHeap::new(); // furthest result on top

        for &ep_id in ep {
            let dist = self.distance(query, ep_id)?;
            visited.insert(ep_id);
            candidates.push(Neighbor::new(ep_id, dist));
            results.push(Neighbor::new(ep_id, dist));
        }

        while let Some(c) = candidates.pop() {
            // If the closest candidate is further than the furthest result, stop
            let furthest_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);
            if c.distance > furthest_dist {
                break;
            }

            // Explore neighbors of c at this layer
            if let Some(node) = &self.nodes[c.id] {
                if layer < node.neighbors.len() {
                    for &neighbor_id in &node.neighbors[layer] {
                        if visited.contains(&neighbor_id) {
                            continue;
                        }
                        visited.insert(neighbor_id);

                        // Skip deleted nodes
                        if self.nodes.get(neighbor_id).and_then(|n| n.as_ref()).is_none() {
                            continue;
                        }

                        let dist = self.distance(query, neighbor_id)?;
                        let furthest_dist =
                            results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                        if dist < furthest_dist || results.len() < ef {
                            candidates.push(Neighbor::new(neighbor_id, dist));
                            results.push(Neighbor::new(neighbor_id, dist));
                            if results.len() > ef {
                                results.pop(); // remove furthest
                            }
                        }
                    }
                }
            }
        }

        Ok(results.into_sorted_vec())
    }

    /// Select the M closest neighbors from candidates (simple selection, Algorithm 3).
    fn select_neighbors_simple(candidates: &[Neighbor], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|n| n.id).collect()
    }

    /// Prune a node's neighbor list at a given layer to at most `m` neighbors.
    fn prune_neighbors(&mut self, node_id: usize, layer: usize, m: usize) {
        // Collect the neighbor IDs and the node's vector
        let (neighbor_ids, node_vec) = {
            let node = match &self.nodes[node_id] {
                Some(n) => n,
                None => return,
            };
            if layer >= node.neighbors.len() {
                return;
            }
            (node.neighbors[layer].clone(), node.vector.clone())
        };

        // Score each neighbor by distance
        let mut scored: Vec<(usize, f32)> = neighbor_ids
            .into_iter()
            .filter_map(|nid| {
                self.nodes.get(nid).and_then(|n| n.as_ref()).map(|n| {
                    let dist = self
                        .metric
                        .distance(&node_vec, &n.vector)
                        .unwrap_or(f32::MAX);
                    (nid, dist)
                })
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(m);

        if let Some(node) = &mut self.nodes[node_id] {
            if layer < node.neighbors.len() {
                node.neighbors[layer] = scored.into_iter().map(|(nid, _)| nid).collect();
            }
        }
    }

    /// INSERT: Algorithm 1 from the HNSW paper.
    pub fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        let level = self.random_level();

        // Ensure the nodes Vec is large enough
        if id >= self.nodes.len() {
            self.nodes.resize_with(id + 1, || None);
        }

        // Create the node
        let node = HnswNode {
            id,
            vector: vector.clone(),
            neighbors: vec![Vec::new(); level + 1],
            level,
        };
        self.nodes[id] = Some(node);
        self.count += 1;

        // If this is the first node, set it as entry point
        let entry_point = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_level = level;
                return Ok(());
            }
            Some(ep) => ep,
        };

        let mut ep_id = entry_point;
        let current_max_level = self.max_level;

        // Phase 1: Greedy descent from top layer down to level+1 (ef=1)
        if current_max_level > level {
            for l in (level + 1..=current_max_level).rev() {
                let nearest = self.search_layer(&vector, &[ep_id], 1, l)?;
                if let Some(n) = nearest.first() {
                    ep_id = n.id;
                }
            }
        }

        // Phase 2: Insert at layers min(level, current_max_level) down to 0
        let insert_from = level.min(current_max_level);
        for l in (0..=insert_from).rev() {
            let m = if l == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };

            let nearest =
                self.search_layer(&vector, &[ep_id], self.params.ef_construction, l)?;

            // Select M closest neighbors
            let neighbors = Self::select_neighbors_simple(&nearest, m);

            // Set the neighbors for this node at this layer
            if let Some(node) = &mut self.nodes[id] {
                if l < node.neighbors.len() {
                    node.neighbors[l] = neighbors.clone();
                }
            }

            // Add bidirectional connections
            for &neighbor_id in &neighbors {
                // First, add the connection and check if pruning is needed
                let needs_pruning = if let Some(neighbor_node) = &mut self.nodes[neighbor_id]
                {
                    if l < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[l].push(id);
                        neighbor_node.neighbors[l].len() > m
                    } else {
                        false
                    }
                } else {
                    false
                };

                // If over capacity, prune in a separate step to avoid borrow conflicts
                if needs_pruning {
                    self.prune_neighbors(neighbor_id, l, m);
                }
            }

            // Update ep for next layer
            if let Some(n) = nearest.first() {
                ep_id = n.id;
            }
        }

        // Update entry point if new node has a higher level
        if level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }

        Ok(())
    }

    /// Remove a node from the graph (lazy deletion — removes from neighbor lists).
    pub fn remove(&mut self, id: usize) -> Result<()> {
        if id >= self.nodes.len() || self.nodes[id].is_none() {
            return Ok(());
        }

        // Remove this node's ID from all its neighbors' neighbor lists
        if let Some(node) = self.nodes[id].take() {
            for (layer, neighbors) in node.neighbors.iter().enumerate() {
                for &neighbor_id in neighbors {
                    if let Some(Some(neighbor_node)) = self.nodes.get_mut(neighbor_id) {
                        if layer < neighbor_node.neighbors.len() {
                            neighbor_node.neighbors[layer].retain(|&n| n != id);
                        }
                    }
                }
            }
            self.count -= 1;

            // Update entry point if we removed it
            if self.entry_point == Some(id) {
                self.entry_point = self
                    .nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(i, n)| n.as_ref().map(|n| (i, n.level)))
                    .max_by_key(|&(_, level)| level)
                    .map(|(i, _)| i);

                self.max_level = self
                    .entry_point
                    .and_then(|ep| self.nodes[ep].as_ref().map(|n| n.level))
                    .unwrap_or(0);
            }
        }

        Ok(())
    }

    /// SEARCH: Algorithm 5 from the HNSW paper.
    ///
    /// Search for the k nearest neighbors, using ef candidates.
    pub fn search_knn(
        &self,
        query: &Vector,
        k: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>> {
        let entry_point = match self.entry_point {
            Some(ep) => ep,
            None => return Ok(vec![]),
        };

        let mut ep_id = entry_point;

        // Phase 1: Greedy descent from top layer to layer 1 (ef=1)
        for l in (1..=self.max_level).rev() {
            let nearest = self.search_layer(query, &[ep_id], 1, l)?;
            if let Some(n) = nearest.first() {
                ep_id = n.id;
            }
        }

        // Phase 2: Search layer 0 with max(ef, k) candidates
        let ef_actual = ef.max(k);
        let mut results = self.search_layer(query, &[ep_id], ef_actual, 0)?;

        // Return top k
        results.truncate(k);
        Ok(results)
    }

    /// Search with a specific ef_search value (runtime tuning without rebuilding).
    pub fn search_with_ef(
        &self,
        query: &Vector,
        k: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>> {
        self.search_knn(query, k, ef)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params() -> HnswParams {
        HnswParams::new(4, 32, 16)
    }

    #[test]
    fn test_insert_single() {
        let mut graph = HnswGraph::new(DistanceMetric::Euclidean, make_params());
        graph
            .insert(0, Vector::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(graph.len(), 1);
        assert!(graph.entry_point.is_some());
    }

    #[test]
    fn test_insert_multiple() {
        let mut graph = HnswGraph::new(DistanceMetric::Euclidean, make_params());
        for i in 0..10 {
            graph
                .insert(i, Vector::new(vec![i as f32, 0.0, 0.0]))
                .unwrap();
        }
        assert_eq!(graph.len(), 10);
    }

    #[test]
    fn test_self_search() {
        let mut graph = HnswGraph::new(DistanceMetric::Euclidean, make_params());
        let vectors: Vec<Vector> = (0..100)
            .map(|i| {
                Vector::new(vec![
                    (i as f32) * 0.1,
                    ((i * 7) as f32) * 0.1,
                    ((i * 13) as f32) * 0.1,
                ])
            })
            .collect();

        for (i, v) in vectors.iter().enumerate() {
            graph.insert(i, v.clone()).unwrap();
        }

        // Search for each inserted vector — the top result should be itself (distance ~0)
        for (i, v) in vectors.iter().enumerate() {
            let results = graph.search_knn(v, 1, 16).unwrap();
            assert!(!results.is_empty(), "No results for vector {}", i);
            assert!(
                results[0].distance < 1e-5,
                "Self-search for {} returned distance {} (id={})",
                i,
                results[0].distance,
                results[0].id
            );
        }
    }

    #[test]
    fn test_search_knn() {
        let mut graph = HnswGraph::new(DistanceMetric::Euclidean, make_params());
        graph.insert(0, Vector::new(vec![0.0, 0.0])).unwrap();
        graph.insert(1, Vector::new(vec![1.0, 0.0])).unwrap();
        graph.insert(2, Vector::new(vec![2.0, 0.0])).unwrap();
        graph.insert(3, Vector::new(vec![3.0, 0.0])).unwrap();
        graph.insert(4, Vector::new(vec![4.0, 0.0])).unwrap();

        let query = Vector::new(vec![0.5, 0.0]);
        let results = graph.search_knn(&query, 2, 16).unwrap();

        assert_eq!(results.len(), 2);
        // The two closest should be id=0 (dist 0.5) and id=1 (dist 0.5)
        let ids: HashSet<usize> = results.iter().map(|n| n.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
    }

    #[test]
    fn test_remove() {
        let mut graph = HnswGraph::new(DistanceMetric::Euclidean, make_params());
        graph.insert(0, Vector::new(vec![1.0, 0.0])).unwrap();
        graph.insert(1, Vector::new(vec![0.0, 1.0])).unwrap();
        assert_eq!(graph.len(), 2);

        graph.remove(0).unwrap();
        assert_eq!(graph.len(), 1);

        let results = graph
            .search_knn(&Vector::new(vec![0.0, 1.0]), 1, 16)
            .unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_remove_entry_point() {
        let mut graph = HnswGraph::new(DistanceMetric::Euclidean, make_params());
        graph.insert(0, Vector::new(vec![1.0, 0.0])).unwrap();
        graph.insert(1, Vector::new(vec![0.0, 1.0])).unwrap();
        graph.insert(2, Vector::new(vec![1.0, 1.0])).unwrap();

        let ep = graph.entry_point.unwrap();
        graph.remove(ep).unwrap();
        assert_eq!(graph.len(), 2);

        // Should still be able to search
        let results = graph
            .search_knn(&Vector::new(vec![0.0, 1.0]), 1, 16)
            .unwrap();
        assert!(!results.is_empty());
    }
}
