//! HTTP route handlers for the vector database API.

use crate::index::Index;
use crate::server::AppState;
use crate::storage::{BatchInsertItem, Metadata, MetadataFilter};
use crate::vector::Vector;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// --- Request/Response types ---

#[derive(Deserialize)]
pub struct InsertRequest {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: Option<usize>,
    #[serde(default)]
    pub filter: Option<MetadataFilter>,
}

#[derive(Deserialize)]
pub struct BatchInsertRequest {
    pub vectors: Vec<BatchInsertItemRequest>,
}

#[derive(Deserialize)]
pub struct BatchInsertItemRequest {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
pub struct BatchSearchRequest {
    pub queries: Vec<BatchSearchQuery>,
    #[serde(default)]
    pub filter: Option<MetadataFilter>,
}

#[derive(Deserialize)]
pub struct BatchSearchQuery {
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
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
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
        .route("/vectors", post(insert_vector::<I>).get(list_vectors::<I>))
        .route(
            "/vectors/batch",
            post(batch_insert::<I>),
        )
        .route(
            "/vectors/:id",
            get(get_vector::<I>).delete(delete_vector::<I>),
        )
        .route("/search", post(search_vectors::<I>))
        .route("/search/batch", post(batch_search::<I>))
        .route("/health", get(health::<I>))
        .route("/metrics", get(get_metrics::<I>))
        .with_state(state)
}

fn hashmap_to_metadata(map: Option<HashMap<String, String>>) -> Metadata {
    let mut meta = Metadata::new();
    if let Some(fields) = map {
        for (k, v) in fields {
            meta.insert(k, v);
        }
    }
    meta
}

// --- Handlers ---

async fn insert_vector<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
    Json(req): Json<InsertRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<ErrorResponse>)> {
    let vector = Vector::new(req.vector);
    let metadata = hashmap_to_metadata(req.metadata);

    let mut store = state.store.write().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    store
        .insert_with_metadata(req.id.clone(), vector, metadata)
        .map_err(|e| {
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

    let vector = store.get(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Vector not found: {}", id),
            }),
        )
    })?;

    let metadata = store
        .get_metadata(&id)
        .map(|m| m.fields().clone())
        .unwrap_or_default();

    Ok(Json(VectorResponse {
        dimension: vector.dimension(),
        vector: vector.as_slice().to_vec(),
        id,
        metadata,
    }))
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

    let results = if let Some(filter) = &req.filter {
        store.search_with_filter(&query, k, filter)
    } else {
        store.search(&query, k)
    }
    .map_err(|e| {
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

async fn batch_insert<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
    Json(req): Json<BatchInsertRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<ErrorResponse>)> {
    let items: Vec<BatchInsertItem> = req
        .vectors
        .into_iter()
        .map(|item| BatchInsertItem {
            id: item.id,
            vector: Vector::new(item.vector),
            metadata: hashmap_to_metadata(item.metadata),
        })
        .collect();

    let count = items.len();

    let mut store = state.store.write().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    store.insert_batch(items).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if let Ok(mut metrics) = state.metrics.write() {
        for _ in 0..count {
            metrics.record_insert();
        }
    }

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({"inserted": count})),
    ))
}

async fn batch_search<I: Index + Send + Sync + std::fmt::Debug + 'static>(
    State(state): State<Arc<AppState<I>>>,
    Json(req): Json<BatchSearchRequest>,
) -> Result<Json<Vec<Vec<SearchResultResponse>>>, (StatusCode, Json<ErrorResponse>)> {
    let queries: Vec<(Vector, usize)> = req
        .queries
        .iter()
        .map(|q| (Vector::new(q.vector.clone()), q.k.unwrap_or(10)))
        .collect();

    let start = Instant::now();

    let store = state.store.read().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Lock poisoned".to_string(),
            }),
        )
    })?;

    let all_results = if let Some(filter) = &req.filter {
        store.search_batch_with_filter(&queries, filter)
    } else {
        store.search_batch(&queries)
    }
    .map_err(|e| {
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

    let response: Vec<Vec<SearchResultResponse>> = all_results
        .into_iter()
        .map(|results| {
            results
                .into_iter()
                .map(|r| SearchResultResponse {
                    id: r.id,
                    distance: r.distance,
                })
                .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat_index::FlatIndex;
    use crate::metrics::MetricsCollector;
    use crate::storage::VectorStore;
    use crate::distance::DistanceMetric;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::RwLock;
    use tower::ServiceExt;

    fn test_app() -> (Router, Arc<AppState<FlatIndex>>) {
        let store = VectorStore::new(DistanceMetric::Euclidean);
        let state = Arc::new(AppState {
            store: RwLock::new(store),
            metrics: RwLock::new(MetricsCollector::new()),
        });
        let app = create_router(state.clone());
        (app, state)
    }

    async fn body_to_json(body: Body) -> serde_json::Value {
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_insert_with_metadata() {
        let (app, _) = test_app();

        let req = Request::builder()
            .method("POST")
            .uri("/vectors")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "id": "v1",
                    "vector": [1.0, 2.0, 3.0],
                    "metadata": {"color": "red", "size": "large"}
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_insert_without_metadata_backward_compat() {
        let (app, _) = test_app();

        let req = Request::builder()
            .method("POST")
            .uri("/vectors")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "id": "v1",
                    "vector": [1.0, 2.0, 3.0]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_get_vector_returns_data() {
        let (app, state) = test_app();

        // Insert a vector with metadata
        {
            let mut store = state.store.write().unwrap();
            let mut meta = Metadata::new();
            meta.insert("color".to_string(), "red".to_string());
            store
                .insert_with_metadata("v1", Vector::new(vec![1.0, 2.0, 3.0]), meta)
                .unwrap();
        }

        let req = Request::builder()
            .method("GET")
            .uri("/vectors/v1")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_to_json(resp.into_body()).await;
        assert_eq!(body["id"], "v1");
        assert_eq!(body["dimension"], 3);
        assert_eq!(body["vector"], serde_json::json!([1.0, 2.0, 3.0]));
        assert_eq!(body["metadata"]["color"], "red");
    }

    #[tokio::test]
    async fn test_search_with_filter() {
        let (app, state) = test_app();

        {
            let mut store = state.store.write().unwrap();
            let mut m1 = Metadata::new();
            m1.insert("color".to_string(), "red".to_string());
            store
                .insert_with_metadata("v1", Vector::new(vec![1.0, 0.0, 0.0]), m1)
                .unwrap();

            let mut m2 = Metadata::new();
            m2.insert("color".to_string(), "blue".to_string());
            store
                .insert_with_metadata("v2", Vector::new(vec![0.9, 0.1, 0.0]), m2)
                .unwrap();
        }

        let req = Request::builder()
            .method("POST")
            .uri("/search")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "vector": [1.0, 0.0, 0.0],
                    "k": 10,
                    "filter": {"op": "eq", "field": "color", "value": "red"}
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_to_json(resp.into_body()).await;
        let results = body.as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["id"], "v1");
    }

    #[tokio::test]
    async fn test_search_without_filter_backward_compat() {
        let (app, state) = test_app();

        {
            let mut store = state.store.write().unwrap();
            store
                .insert("v1", Vector::new(vec![1.0, 0.0, 0.0]))
                .unwrap();
        }

        let req = Request::builder()
            .method("POST")
            .uri("/search")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "vector": [1.0, 0.0, 0.0],
                    "k": 10
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_to_json(resp.into_body()).await;
        let results = body.as_array().unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_batch_insert_endpoint() {
        let (app, state) = test_app();

        let req = Request::builder()
            .method("POST")
            .uri("/vectors/batch")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "vectors": [
                        {"id": "v1", "vector": [1.0, 0.0, 0.0]},
                        {"id": "v2", "vector": [0.0, 1.0, 0.0], "metadata": {"color": "blue"}}
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let body = body_to_json(resp.into_body()).await;
        assert_eq!(body["inserted"], 2);

        let store = state.store.read().unwrap();
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn test_batch_search_endpoint() {
        let (app, state) = test_app();

        {
            let mut store = state.store.write().unwrap();
            store
                .insert("v1", Vector::new(vec![1.0, 0.0, 0.0]))
                .unwrap();
            store
                .insert("v2", Vector::new(vec![0.0, 1.0, 0.0]))
                .unwrap();
        }

        let req = Request::builder()
            .method("POST")
            .uri("/search/batch")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "queries": [
                        {"vector": [1.0, 0.0, 0.0], "k": 1},
                        {"vector": [0.0, 1.0, 0.0], "k": 1}
                    ]
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_to_json(resp.into_body()).await;
        let results = body.as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0]["id"], "v1");
        assert_eq!(results[1][0]["id"], "v2");
    }

    #[tokio::test]
    async fn test_batch_search_with_filter_endpoint() {
        let (app, state) = test_app();

        {
            let mut store = state.store.write().unwrap();
            let mut m1 = Metadata::new();
            m1.insert("color".to_string(), "red".to_string());
            store
                .insert_with_metadata("v1", Vector::new(vec![1.0, 0.0, 0.0]), m1)
                .unwrap();

            let mut m2 = Metadata::new();
            m2.insert("color".to_string(), "blue".to_string());
            store
                .insert_with_metadata("v2", Vector::new(vec![0.0, 1.0, 0.0]), m2)
                .unwrap();
        }

        let req = Request::builder()
            .method("POST")
            .uri("/search/batch")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "queries": [
                        {"vector": [1.0, 0.0, 0.0], "k": 10},
                        {"vector": [0.0, 1.0, 0.0], "k": 10}
                    ],
                    "filter": {"op": "eq", "field": "color", "value": "red"}
                })
                .to_string(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = body_to_json(resp.into_body()).await;
        let results = body.as_array().unwrap();
        assert_eq!(results.len(), 2);
        // Both queries should only return v1 (the red one)
        assert_eq!(results[0].as_array().unwrap().len(), 1);
        assert_eq!(results[0][0]["id"], "v1");
        assert_eq!(results[1].as_array().unwrap().len(), 1);
        assert_eq!(results[1][0]["id"], "v1");
    }
}
