# VectorDB From Scratch

A vector database implementation in Rust, built from scratch as a learning project to understand database internals and systems programming.

## Features

- **Vector storage** with CRUD operations and string-based IDs
- **Distance metrics**: Euclidean, Cosine, Dot Product
- **Brute-force search** (FlatIndex) and **approximate nearest neighbor** search (HNSW)
- **Metadata filtering** with composable filter expressions (eq, ne, exists, and, or)
- **Batch operations** for bulk inserts and parallel searches
- **Persistence** with write-ahead log (WAL), snapshots, and crash recovery
- **HTTP API** (9 endpoints) powered by Axum
- **Metrics collection** with latency percentiles and operation counters
- **CLI** for direct interaction and running the HTTP server
- **89 tests** — unit, integration, recall, and doc tests

## Installation

This project uses [Nix](https://nixos.org/) with flakes for reproducible development environments.

### Prerequisites
1. Install Nix with flakes enabled
2. Clone this repository

### Enter Development Environment

```bash
# Enter the Nix development shell
nix develop

# Or use direnv for automatic activation
echo "use flake" > .envrc
direnv allow
```

The Nix shell provides:
- Rust toolchain (rustc, cargo, clippy, rustfmt)
- Development tools (cargo-watch, cargo-edit, cargo-nextest, cargo-criterion)

## Usage

### Build

```bash
cargo build --release
```

### Run Tests

```bash
# Run unit tests
cargo test

# Run with nextest (faster)
cargo nextest run
```

### CLI

```bash
# Insert a vector
cargo run -- insert v1 --vector "1.0,2.0,3.0"

# Search for similar vectors (default k=5)
cargo run -- search "1.1,2.1,3.1" --k 10

# Delete a vector
cargo run -- delete v1

# List all vector IDs
cargo run -- list

# Use HNSW index instead of brute-force
cargo run -- --index hnsw insert v1 --vector "1.0,2.0,3.0"

# Enable persistence with a data directory
cargo run -- --data-dir ./db insert v1 --vector "1.0,2.0,3.0"

# Start the HTTP API server (default: 0.0.0.0:3000)
cargo run -- serve
cargo run -- serve --addr 127.0.0.1:8080
```

### HTTP API

Start the server with `cargo run -- serve`, then interact via HTTP:

#### Insert a vector

```bash
curl -X POST http://localhost:3000/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "v1", "vector": [1.0, 2.0, 3.0], "metadata": {"color": "red"}}'
```

#### Batch insert

```bash
curl -X POST http://localhost:3000/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{"vectors": [
    {"id": "v1", "vector": [1.0, 2.0, 3.0]},
    {"id": "v2", "vector": [4.0, 5.0, 6.0], "metadata": {"color": "blue"}}
  ]}'
```

#### Search

```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [1.1, 2.1, 3.1], "k": 5}'
```

#### Search with metadata filter

```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [1.1, 2.1, 3.1], "k": 5, "filter": {"op": "eq", "field": "color", "value": "red"}}'
```

#### Batch search

```bash
curl -X POST http://localhost:3000/search/batch \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"vector": [1.0, 2.0, 3.0], "k": 3}, {"vector": [4.0, 5.0, 6.0], "k": 3}]}'
```

#### Other endpoints

```bash
# List all vector IDs
curl http://localhost:3000/vectors

# Get a specific vector
curl http://localhost:3000/vectors/v1

# Delete a vector
curl -X DELETE http://localhost:3000/vectors/v1

# Health check
curl http://localhost:3000/health

# Metrics
curl http://localhost:3000/metrics
```

### API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/vectors` | Insert a vector (with optional metadata) |
| `GET` | `/vectors` | List all vector IDs |
| `GET` | `/vectors/:id` | Get a vector by ID |
| `DELETE` | `/vectors/:id` | Delete a vector |
| `POST` | `/vectors/batch` | Batch insert vectors |
| `POST` | `/search` | Search for similar vectors (with optional filter) |
| `POST` | `/search/batch` | Batch search queries |
| `GET` | `/health` | Health check with vector count |
| `GET` | `/metrics` | Query latency percentiles and operation counters |

### Metadata Filters

Filters are composable JSON expressions using a tagged `"op"` field:

```json
{"op": "eq", "field": "color", "value": "red"}
{"op": "ne", "field": "color", "value": "blue"}
{"op": "exists", "field": "category"}
{"op": "and", "filters": [
  {"op": "eq", "field": "color", "value": "red"},
  {"op": "exists", "field": "category"}
]}
{"op": "or", "filters": [
  {"op": "eq", "field": "color", "value": "red"},
  {"op": "eq", "field": "color", "value": "blue"}
]}
```

### Benchmarks

```bash
# Run benchmarks (FlatIndex and HNSW comparison)
cargo bench

# Results will be in target/criterion/
```

## Architecture

### Index Types

- **FlatIndex** — Brute-force O(n) search. Exact results, simple and reliable.
- **HnswIndex** — Approximate nearest neighbor search using [Hierarchical Navigable Small World](https://arxiv.org/abs/1603.09320) graphs. Achieves >95% recall with significantly faster search on large datasets.

HNSW default parameters: `m=16`, `ef_construction=200`, `ef_search=50`, `max_layers=16`.

### Persistence

- **Write-Ahead Log (WAL)** — All inserts and deletes are durably logged before being applied. Entries are length-prefixed bincode with CRC32 checksums.
- **Snapshots** — Periodic checkpoints of the full dataset (default: every 1,000 WAL entries).
- **Crash Recovery** — On startup, loads the latest snapshot and replays any WAL entries written after it.
- **Memory-mapped I/O** — Optional mmap-based reads for snapshot files.

### Metrics

The `/metrics` endpoint reports:
- Total queries, inserts, and deletes
- Average, p50, p95, and p99 query latency (microseconds)

## Project Structure

```
vectordb-from-scratch/
├── flake.nix                    # Nix development environment
├── Cargo.toml                   # Rust dependencies
├── src/
│   ├── lib.rs                   # Library entry point, public API
│   ├── main.rs                  # CLI application
│   ├── vector.rs                # Vector type and operations
│   ├── storage.rs               # VectorStore<I: Index>, metadata, search
│   ├── distance.rs              # Distance metrics
│   ├── index.rs                 # Index trait (abstract interface)
│   ├── flat_index.rs            # Brute-force index
│   ├── error.rs                 # Error types
│   ├── metrics.rs               # Latency percentiles and counters
│   ├── hnsw/
│   │   ├── mod.rs               # HnswIndex public API
│   │   ├── graph.rs             # HNSW graph and algorithm
│   │   └── neighbor_queue.rs    # Priority queue helpers
│   ├── persistence/
│   │   ├── mod.rs               # Module exports
│   │   ├── engine.rs            # StorageEngine (WAL + snapshots)
│   │   ├── wal.rs               # Write-ahead log
│   │   ├── snapshot.rs          # Snapshot manager
│   │   ├── serialization.rs     # Bincode + CRC32 serialization
│   │   └── mmap.rs              # Memory-mapped file I/O
│   └── server/
│       ├── mod.rs               # Server startup
│       └── routes.rs            # HTTP endpoint handlers
├── tests/
│   ├── integration_test.rs      # End-to-end workflow tests
│   └── recall_test.rs           # HNSW recall benchmarks
└── benches/
    ├── search_bench.rs          # FlatIndex benchmarks
    └── hnsw_bench.rs            # HNSW vs FlatIndex benchmarks
```

## Development Roadmap

- [x] **Phase 1 — Foundation**: Vector type, distance metrics, VectorStore, CLI, tests
- [x] **Phase 2 — Indexing**: Index trait, FlatIndex extraction, HNSW index
- [x] **Phase 3 — Persistence**: WAL, snapshots, crash recovery, mmap storage
- [x] **Phase 4 — HTTP API**: Axum server, metrics, parallel HNSW, CLI serve command
- [x] **Phase 5 — Search Enhancements**: Metadata-filtered search, batch operations, API enhancements

## Learning Resources

This project follows the tutorial:
- [Write You a Vector Database](https://skyzh.github.io/write-you-a-vector-db/) by Alex Chi

Key academic papers:
- [HNSW (2016)](https://arxiv.org/abs/1603.09320) - Hierarchical Navigable Small World graphs
- [Faiss (2024)](https://arxiv.org/abs/2401.08281) - Facebook's billion-scale similarity search

Reference implementations:
- [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)
- [pgvecto.rs](https://github.com/tensorchord/pgvecto.rs)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Ricardo Ledan
