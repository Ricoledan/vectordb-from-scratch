//! Benchmarks for vector search

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vectordb_from_scratch::{DistanceMetric, Vector, VectorStore};

fn create_random_vectors(n: usize, dim: usize) -> Vec<Vector> {
    (0..n)
        .map(|_| {
            let data: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            Vector::new(data)
        })
        .collect()
}

fn benchmark_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    for size in [100, 1000, 10000].iter() {
        let mut store = VectorStore::new(DistanceMetric::Euclidean);
        let vectors = create_random_vectors(*size, 128);

        // Insert vectors
        for (i, v) in vectors.iter().enumerate() {
            store.insert(format!("v{}", i), v.clone()).unwrap();
        }

        let query = Vector::new(vec![0.5; 128]);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                store.search(black_box(&query), black_box(10)).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_search);
criterion_main!(benches);
