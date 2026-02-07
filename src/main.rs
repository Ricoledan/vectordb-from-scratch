//! CLI interface for the vector database

use anyhow::Result;
use clap::{Parser, Subcommand};
use vectordb_from_scratch::{DistanceMetric, Vector, VectorStore};

#[derive(Parser)]
#[command(name = "vectordb")]
#[command(about = "A vector database built from scratch in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // For this basic version, we create an in-memory store
    // In future versions, this will load from persistence
    let mut store = VectorStore::new(DistanceMetric::Euclidean);

    match cli.command {
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
    }

    Ok(())
}
