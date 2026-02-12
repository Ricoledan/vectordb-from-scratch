//! CLI interface for the vector database

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use vectordb_from_scratch::persistence::engine::{EngineConfig, StorageEngine};
use vectordb_from_scratch::{
    DistanceMetric, HnswIndex, HnswParams, Index, Vector, VectorStore,
};

#[derive(Parser)]
#[command(name = "vectordb")]
#[command(about = "A vector database built from scratch in Rust", long_about = None)]
struct Cli {
    /// Index type to use for search
    #[arg(long, value_enum, default_value = "flat")]
    index: IndexType,

    /// Data directory for persistence. If set, data is persisted to disk.
    #[arg(long)]
    data_dir: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(ValueEnum, Clone)]
enum IndexType {
    Flat,
    Hnsw,
}

#[derive(Subcommand)]
enum Commands {
    /// Insert a vector
    Insert {
        /// Vector ID
        id: String,
        /// Vector data as comma-separated values (e.g., "1.0,2.0,3.0")
        #[arg(short, long)]
        vector: String,
    },
    /// Search for similar vectors
    Search {
        /// Query vector as comma-separated values (e.g., "1.0,2.0,3.0")
        query: String,
        /// Number of results to return
        #[arg(short, long, default_value = "5")]
        k: usize,
    },
    /// Delete a vector
    Delete {
        /// Vector ID to delete
        id: String,
    },
    /// List all vector IDs
    List,
    /// Start the HTTP API server
    Serve {
        /// Address to bind to
        #[arg(long, default_value = "0.0.0.0:3000")]
        addr: String,
    },
}

fn run_with_engine(mut engine: StorageEngine, command: Commands) -> Result<()> {
    match command {
        Commands::Insert { id, vector } => {
            let v = Vector::from_str(&vector)?;
            engine.insert(id.clone(), v)?;
            println!("Inserted vector with ID: {}", id);
        }
        Commands::Search { query, k } => {
            let q = Vector::from_str(&query)?;
            let results = engine.search(&q, k)?;

            if results.is_empty() {
                println!("No results found (store is empty)");
            } else {
                println!("Top {} results:", results.len());
                for (i, result) in results.iter().enumerate() {
                    println!("{}. {} (distance: {:.4})", i + 1, result.id, result.distance);
                }
            }
        }
        Commands::Delete { id } => {
            engine.delete(&id)?;
            println!("Deleted vector with ID: {}", id);
        }
        Commands::List => {
            let ids = engine.list_ids();
            if ids.is_empty() {
                println!("No vectors in store");
            } else {
                println!("Vector IDs ({} total):", ids.len());
                for id in ids {
                    println!("  - {}", id);
                }
            }
        }
        Commands::Serve { .. } => {
            anyhow::bail!("Serve command is not supported with --data-dir (persistent storage). Use in-memory mode.");
        }
    }
    Ok(())
}

fn run_in_memory<I: Index + std::fmt::Debug>(
    mut store: VectorStore<I>,
    command: Commands,
) -> Result<()> {
    match command {
        Commands::Insert { id, vector } => {
            let v = Vector::from_str(&vector)?;
            store.insert(id.clone(), v)?;
            println!("Inserted vector with ID: {}", id);
        }
        Commands::Search { query, k } => {
            let q = Vector::from_str(&query)?;
            let results = store.search(&q, k)?;

            if results.is_empty() {
                println!("No results found (store is empty)");
            } else {
                println!("Top {} results:", results.len());
                for (i, result) in results.iter().enumerate() {
                    println!("{}. {} (distance: {:.4})", i + 1, result.id, result.distance);
                }
            }
        }
        Commands::Delete { id } => {
            store.delete(&id)?;
            println!("Deleted vector with ID: {}", id);
        }
        Commands::List => {
            let ids = store.list_ids();
            if ids.is_empty() {
                println!("No vectors in store");
            } else {
                println!("Vector IDs ({} total):", ids.len());
                for id in ids {
                    println!("  - {}", id);
                }
            }
        }
        Commands::Serve { .. } => {
            unreachable!("Serve handled separately");
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle serve command specially — it needs the async runtime
    if let Commands::Serve { ref addr } = cli.command {
        let metric = DistanceMetric::Euclidean;
        match cli.index {
            IndexType::Flat => {
                vectordb_from_scratch::server::start_flat(addr, metric).await?;
            }
            IndexType::Hnsw => {
                vectordb_from_scratch::server::start_hnsw(
                    addr,
                    metric,
                    HnswParams::default(),
                )
                .await?;
            }
        }
        return Ok(());
    }

    // If --data-dir is set, use persistent storage engine
    if let Some(data_dir) = cli.data_dir {
        let config = EngineConfig {
            checkpoint_interval: 1000,
            metric: DistanceMetric::Euclidean,
        };
        let engine = StorageEngine::open(data_dir, config)?;
        return run_with_engine(engine, cli.command);
    }

    // Otherwise, in-memory
    match cli.index {
        IndexType::Flat => {
            let store = VectorStore::with_flat_index(DistanceMetric::Euclidean);
            run_in_memory(store, cli.command)
        }
        IndexType::Hnsw => {
            let index =
                HnswIndex::with_params(DistanceMetric::Euclidean, HnswParams::default());
            let store = VectorStore::with_index(index);
            run_in_memory(store, cli.command)
        }
    }
}
