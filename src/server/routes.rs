//! HTTP route handlers for the vector database API.

use crate::index::Index;
use crate::server::AppState;
use crate::vector::Vector;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

// --- Request/Response types ---

#[derive(Deserialize)]
pub struct InsertRequest {
    pub id: String,
    pub vector: Vec<f32>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: Option<usize>,
}

#[derive(Serialize)]
pub struct SearchResultResponse {
    pub id: String,
    pub distance: f32,
}

#[derive(Serialize)]
pub struct VectorResponse {
    pub id: String,
    pub dimension: usize,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub vector_count: usize,
}

#[derive(Serialize)]
pub struct MetricsResponse {
    pub total_queries: u64,
    pub total_inserts: u64,
    pub total_deletes: u64,
    pub avg_query_latency_us: f64,
    pub p50_query_latency_us: f64,
    pub p95_query_latency_us: f64,
    pub p99_query_latency_us: f64,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// --- Router ---

pub fn create_router<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    state: Arc<AppState<I>>,
) -> Router {
    Router::new()
        .route("/vectors", post(insert_vector::<I>))
        .route("/vectors", get(list_vectors::<I>))
        .route("/vectors/{id}", get(get_vector::<I>))
        .route("/vectors/{id}", delete(delete_vector::<I>))
        .route("/search", post(search_vectors::<I>))
        .route("/health", get(health::<I>))
        .route("/metrics", get(get_metrics::<I>))
        .with_state(state)
}

// --- Handlers ---

async fn insert_vector<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
    Json(req): Json<InsertRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<ErrorResponse>)> {
    let vector = Vector::new(req.vector);

    let mut store = state.store.write().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    store.insert(req.id.clone(), vector).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if let Ok(mut metrics) = state.metrics.write() {
        metrics.record_insert();
    }

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({"id": req.id, "status": "inserted"})),
    ))
}

async fn get_vector<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
    Path(id): Path<String>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let store = state.store.read().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    let ids = store.list_ids();
    if ids.contains(&id) {
        Ok(Json(VectorResponse {
            dimension: store.dimension().unwrap_or(0),
            id,
        }))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Vector not found: {}", id),
            }),
        ))
    }
}

async fn delete_vector<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let mut store = state.store.write().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    store.delete(&id).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if let Ok(mut metrics) = state.metrics.write() {
        metrics.record_delete();
    }

    Ok(Json(serde_json::json!({"id": id, "status": "deleted"})))
}

async fn search_vectors<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<SearchResultResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let query = Vector::new(req.vector);
    let k = req.k.unwrap_or(10);

    let start = Instant::now();

    let store = state.store.read().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    let results = store.search(&query, k).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let elapsed = start.elapsed();

    if let Ok(mut metrics) = state.metrics.write() {
        metrics.record_query(elapsed);
    }

    let response: Vec<SearchResultResponse> = results
        .into_iter()
        .map(|r| SearchResultResponse {
            id: r.id,
            distance: r.distance,
        })
        .collect();

    Ok(Json(response))
}

async fn list_vectors<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    let store = state.store.read().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    Ok(Json(store.list_ids()))
}

async fn health<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
) -> Json<HealthResponse> {
    let count = state
        .store
        .read()
        .map(|s| s.len())
        .unwrap_or(0);

    Json(HealthResponse {
        status: "ok".to_string(),
        vector_count: count,
    })
}

async fn get_metrics<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
) -> Json<MetricsResponse> {
    let metrics = state.metrics.read().unwrap();

    Json(MetricsResponse {
        total_queries: metrics.total_queries(),
        total_inserts: metrics.total_inserts(),
        total_deletes: metrics.total_deletes(),
        avg_query_latency_us: metrics.avg_query_latency_us(),
        p50_query_latency_us: metrics.percentile_query_latency_us(50.0),
        p95_query_latency_us: metrics.percentile_query_latency_us(95.0),
        p99_query_latency_us: metrics.percentile_query_latency_us(99.0),
    })
}
