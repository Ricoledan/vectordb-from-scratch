# VectorDB From Scratch

A vector database implementation in Rust, built from scratch as a learning project to understand database internals and systems programming.

## Features

### Phase 1: Foundation (Current)
- ✅ Vector type with basic operations
- ✅ Distance metrics: Euclidean, Cosine, Dot Product
- ✅ In-memory vector storage with CRUD operations
- ✅ Brute-force k-NN search
- ✅ CLI interface
- ✅ Comprehensive test suite
- ✅ Benchmarking framework

### Planned Features
- **Phase 2**: HNSW (Hierarchical Navigable Small World) index for fast approximate search
- **Phase 3**: Persistence layer with WAL and memory-mapped storage
- **Phase 4**: HTTP/gRPC API, advanced metrics, production optimizations

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
- All dependencies for building the project

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

The database currently operates as an in-memory store. Future versions will add persistence.

```bash
# Insert a vector
cargo run -- insert v1 --vector "1.0,2.0,3.0"

# Search for similar vectors
cargo run -- search "1.1,2.1,3.1" --k 5

# Delete a vector
cargo run -- delete v1

# List all vectors
cargo run -- list
```

### Benchmarks

```bash
# Run benchmarks
cargo bench

# Results will be in target/criterion/
```

## Project Structure

```
vectordb-from-scratch/
├── flake.nix           # Nix development environment
├── Cargo.toml          # Rust dependencies
├── src/
│   ├── lib.rs          # Library entry point
│   ├── main.rs         # CLI application
│   ├── vector.rs       # Vector type and operations
│   ├── storage.rs      # In-memory storage implementation
│   ├── distance.rs     # Distance metrics
│   └── error.rs        # Error types
├── tests/
│   └── integration_test.rs
└── benches/
    └── search_bench.rs
```

## Learning Resources

This project follows the tutorial:
- [Write You a Vector Database](https://skyzh.github.io/write-you-a-vector-db/) by Alex Chi

Key academic papers:
- [HNSW (2016)](https://arxiv.org/abs/1603.09320) - Hierarchical Navigable Small World graphs
- [Faiss (2024)](https://arxiv.org/abs/2401.08281) - Facebook's billion-scale similarity search

Reference implementations:
- [hnswlib-rs](https://github.com/jean-pierreBoth/hnswlib-rs)
- [pgvecto.rs](https://github.com/tensorchord/pgvecto.rs)

## Development Roadmap

### Week 1-2: Foundation ✅
- [x] Project setup with Nix
- [x] Vector type and distance metrics
- [x] In-memory storage
- [x] Brute-force search
- [x] CLI interface
- [x] Tests and benchmarks

### Week 3-4: HNSW Index
- [ ] HNSW graph construction
- [ ] HNSW search algorithm
- [ ] Parameter tuning
- [ ] Recall/performance benchmarks

### Week 5-6: Persistence
- [ ] Write-ahead log (WAL)
- [ ] Memory-mapped storage
- [ ] Crash recovery
- [ ] Durability tests

### Week 7-8: Polish
- [ ] HTTP/gRPC API
- [ ] Advanced distance metrics
- [ ] Comprehensive documentation
- [ ] Production benchmarks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Ricardo Ledan

## Acknowledgments

Built as a learning project following industry best practices and academic research in vector similarity search.
