//! Priority queue utilities for HNSW — handles f32 ordering for BinaryHeap.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A neighbor entry with a distance and internal ID.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub distance: f32,
    pub id: usize,
}

impl Neighbor {
    pub fn new(id: usize, distance: f32) -> Self {
        Self { distance, id }
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for Neighbor {}

// Default ordering: max-heap (largest distance on top).
// We reverse this for min-heap by wrapping in `std::cmp::Reverse` or
// using the `MinHeap` wrapper below.
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// A wrapper that reverses Neighbor ordering to create a min-heap.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Reversed(pub Neighbor);

impl PartialOrd for Reversed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Reversed {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

/// Max-heap of neighbors (largest distance on top). Used as the result set bounded by ef.
pub struct MaxHeap {
    heap: BinaryHeap<Neighbor>,
}

impl MaxHeap {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    pub fn push(&mut self, n: Neighbor) {
        self.heap.push(n);
    }

    /// Push and pop the max if size exceeds limit, keeping only the closest `limit` neighbors.
    pub fn push_bounded(&mut self, n: Neighbor, limit: usize) {
        self.heap.push(n);
        if self.heap.len() > limit {
            self.heap.pop();
        }
    }

    pub fn peek(&self) -> Option<&Neighbor> {
        self.heap.peek()
    }

    pub fn pop(&mut self) -> Option<Neighbor> {
        self.heap.pop()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain into a sorted Vec (ascending by distance).
    pub fn into_sorted_vec(self) -> Vec<Neighbor> {
        let mut v: Vec<Neighbor> = self.heap.into_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        v
    }
}

/// Min-heap of neighbors (smallest distance on top). Used as the candidate set.
pub struct MinHeap {
    heap: BinaryHeap<Reversed>,
}

impl MinHeap {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    pub fn push(&mut self, n: Neighbor) {
        self.heap.push(Reversed(n));
    }

    pub fn peek(&self) -> Option<&Neighbor> {
        self.heap.peek().map(|r| &r.0)
    }

    pub fn pop(&mut self) -> Option<Neighbor> {
        self.heap.pop().map(|r| r.0)
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_heap_ordering() {
        let mut heap = MaxHeap::new();
        heap.push(Neighbor::new(0, 3.0));
        heap.push(Neighbor::new(1, 1.0));
        heap.push(Neighbor::new(2, 2.0));

        assert_eq!(heap.pop().unwrap().distance, 3.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
        assert_eq!(heap.pop().unwrap().distance, 1.0);
    }

    #[test]
    fn test_min_heap_ordering() {
        let mut heap = MinHeap::new();
        heap.push(Neighbor::new(0, 3.0));
        heap.push(Neighbor::new(1, 1.0));
        heap.push(Neighbor::new(2, 2.0));

        assert_eq!(heap.pop().unwrap().distance, 1.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
        assert_eq!(heap.pop().unwrap().distance, 3.0);
    }

    #[test]
    fn test_bounded_push() {
        let mut heap = MaxHeap::new();
        heap.push_bounded(Neighbor::new(0, 5.0), 2);
        heap.push_bounded(Neighbor::new(1, 1.0), 2);
        heap.push_bounded(Neighbor::new(2, 3.0), 2);

        assert_eq!(heap.len(), 2);
        let sorted = heap.into_sorted_vec();
        assert_eq!(sorted[0].distance, 1.0);
        assert_eq!(sorted[1].distance, 3.0);
    }

    #[test]
    fn test_into_sorted_vec() {
        let mut heap = MaxHeap::new();
        heap.push(Neighbor::new(0, 5.0));
        heap.push(Neighbor::new(1, 1.0));
        heap.push(Neighbor::new(2, 3.0));
        heap.push(Neighbor::new(3, 2.0));

        let sorted = heap.into_sorted_vec();
        for i in 0..sorted.len() - 1 {
            assert!(sorted[i].distance <= sorted[i + 1].distance);
        }
    }
}
