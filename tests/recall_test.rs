//! Recall tests: verify HNSW finds a high percentage of true nearest neighbors.

use rand::Rng;
use vectordb_from_scratch::{
    DistanceMetric, FlatIndex, HnswIndex, HnswParams, Index, Vector,
};

fn random_vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let data: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            Vector::new(data)
        })
        .collect()
}

fn recall_at_k(flat_results: &[(usize, f32)], hnsw_results: &[(usize, f32)]) -> f64 {
    let ground_truth: std::collections::HashSet<usize> =
        flat_results.iter().map(|(id, _)| *id).collect();
    let found: usize = hnsw_results
        .iter()
        .filter(|(id, _)| ground_truth.contains(id))
        .count();
    found as f64 / flat_results.len() as f64
}

fn test_recall(n: usize, dim: usize, k: usize, num_queries: usize, min_recall: f64) {
    let vectors = random_vectors(n, dim);

    // Build flat index (ground truth)
    let mut flat = FlatIndex::new(DistanceMetric::Euclidean);
    for (i, v) in vectors.iter().enumerate() {
        flat.add(i, v.clone()).unwrap();
    }

    // Build HNSW index
    let params = HnswParams::new(16, 200, 50);
    let mut hnsw = HnswIndex::with_params(DistanceMetric::Euclidean, params);
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i, v.clone()).unwrap();
    }

    // Run queries
    let queries = random_vectors(num_queries, dim);
    let mut total_recall = 0.0;

    for query in &queries {
        let flat_results = flat.search(query, k).unwrap();
        // Use higher ef for search to improve recall
        let hnsw_results = hnsw.search_with_ef(query, k, 100).unwrap();
        total_recall += recall_at_k(&flat_results, &hnsw_results);
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(
        avg_recall >= min_recall,
        "Recall {:.3} is below threshold {:.3} for n={}, dim={}, k={}",
        avg_recall,
        min_recall,
        n,
        dim,
        k
    );
}

#[test]
fn test_recall_100_vectors() {
    test_recall(100, 32, 10, 50, 0.90);
}

#[test]
fn test_recall_1000_vectors() {
    test_recall(1000, 64, 10, 50, 0.90);
}

#[test]
fn test_recall_5000_vectors() {
    test_recall(5000, 128, 10, 20, 0.85);
}
