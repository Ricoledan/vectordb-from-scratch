//! HNSW vs brute-force benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vectordb_from_scratch::{
    DistanceMetric, FlatIndex, HnswIndex, HnswParams, Index, Vector,
};

fn create_random_vectors(n: usize, dim: usize) -> Vec<Vector> {
    (0..n)
        .map(|_| {
            let data: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            Vector::new(data)
        })
        .collect()
}

fn benchmark_hnsw_vs_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_vs_flat");
    group.sample_size(20);

    for &size in &[1_000, 10_000] {
        let dim = 128;
        let vectors = create_random_vectors(size, dim);
        let query = Vector::new(vec![0.5; dim]);

        // Build flat index
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

        group.bench_with_input(
            BenchmarkId::new("flat", size),
            &size,
            |b, _| {
                b.iter(|| flat.search(black_box(&query), black_box(10)).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hnsw", size),
            &size,
            |b, _| {
                b.iter(|| hnsw.search(black_box(&query), black_box(10)).unwrap());
            },
        );
    }

    group.finish();
}

fn benchmark_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    group.sample_size(10);

    let dim = 128;
    let vectors = create_random_vectors(1_000, dim);

    group.bench_function("insert_1000_128d", |b| {
        b.iter(|| {
            let params = HnswParams::new(16, 200, 50);
            let mut hnsw = HnswIndex::with_params(DistanceMetric::Euclidean, params);
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i, v.clone()).unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_hnsw_vs_flat, benchmark_hnsw_insert);
criterion_main!(benches);
