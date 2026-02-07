//! Integration tests for the vector database

use vectordb_from_scratch::{DistanceMetric, Vector, VectorStore};

#[test]
fn test_basic_workflow() {
    let mut store = VectorStore::new(DistanceMetric::Euclidean);

    // Insert vectors
    store.insert("v1", Vector::new(vec![1.0, 0.0, 0.0])).unwrap();
    store.insert("v2", Vector::new(vec![0.0, 1.0, 0.0])).unwrap();
    store.insert("v3", Vector::new(vec![0.0, 0.0, 1.0])).unwrap();

    // Verify store size
    assert_eq!(store.len(), 3);

    // Search
    let query = Vector::new(vec![1.0, 0.1, 0.0]);
    let results = store.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "v1");

    // Delete
    store.delete("v2").unwrap();
    assert_eq!(store.len(), 2);
}

#[test]
fn test_different_metrics() {
    let metrics = vec![
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ];

    for metric in metrics {
        let mut store = VectorStore::new(metric);
        store.insert("v1", Vector::new(vec![1.0, 2.0, 3.0])).unwrap();

        let query = Vector::new(vec![1.0, 2.0, 3.0]);
        let results = store.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }
}
