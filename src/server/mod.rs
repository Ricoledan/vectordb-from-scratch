//! HTTP API server for the vector database.

pub mod routes;

use crate::distance::DistanceMetric;
use crate::hnsw::{HnswIndex, HnswParams};
use crate::index::Index;
use crate::metrics::MetricsCollector;
use crate::storage::VectorStore;
use std::sync::{Arc, RwLock};

/// Shared application state for the HTTP server.
pub struct AppState<I: Index> {
    pub store: RwLock<VectorStore<I>>,
    pub metrics: RwLock<MetricsCollector>,
}

/// Start the HTTP server with a flat index.
pub async fn start_flat(addr: &str, metric: DistanceMetric) -> anyhow::Result<()> {
    let store = VectorStore::with_flat_index(metric);
    let state = Arc::new(AppState {
        store: RwLock::new(store),
        metrics: RwLock::new(MetricsCollector::new()),
    });

    let app = routes::create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Server listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Start the HTTP server with an HNSW index.
pub async fn start_hnsw(
    addr: &str,
    metric: DistanceMetric,
    params: HnswParams,
) -> anyhow::Result<()> {
    let index = HnswIndex::with_params(metric, params);
    let store = VectorStore::with_index(index);
    let state = Arc::new(AppState {
        store: RwLock::new(store),
        metrics: RwLock::new(MetricsCollector::new()),
    });

    let app = routes::create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Server listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
