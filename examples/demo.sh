#!/usr/bin/env bash
# demo.sh — Interactive demo of the vectordb-from-scratch HTTP API
# Run: bash examples/demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_URL="http://127.0.0.1:3000"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo ""
        echo "=== Cleaning up ==="
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
}
trap cleanup EXIT

# -------------------------------------------------------
echo "============================================"
echo "  Vector Database from Scratch — Demo"
echo "============================================"
echo ""

# Step 1: Build
echo "=== Step 1: Building the project (release mode) ==="
cargo build --release --manifest-path "$PROJECT_ROOT/Cargo.toml"
echo "Build complete."
echo ""

# Step 2: Start server
echo "=== Step 2: Starting the server ==="
BINARY="$PROJECT_ROOT/target/release/vectordb_from_scratch"
"$BINARY" --index flat serve --addr 127.0.0.1:3000 &
SERVER_PID=$!
echo "Server started in background (PID $SERVER_PID)"
echo ""

# Step 3: Wait for health check
echo "=== Step 3: Waiting for server to be ready ==="
for i in $(seq 1 30); do
    if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Server did not become ready in time."
        exit 1
    fi
    sleep 0.2
done
echo ""

# Step 4: Insert vectors with metadata
echo "=== Step 4: Inserting vectors with metadata ==="
echo ""

echo "Inserting 'sunset' (warm colors, reddish direction)..."
curl -s -X POST "$BASE_URL/vectors" \
    -H "Content-Type: application/json" \
    -d '{
        "id": "sunset",
        "vector": [0.9, 0.1, 0.0],
        "metadata": {"color": "red", "category": "nature"}
    }' | jq .
echo ""

echo "Inserting 'ocean' (cool colors, bluish direction)..."
curl -s -X POST "$BASE_URL/vectors" \
    -H "Content-Type: application/json" \
    -d '{
        "id": "ocean",
        "vector": [0.0, 0.2, 0.9],
        "metadata": {"color": "blue", "category": "nature"}
    }' | jq .
echo ""

echo "Inserting 'fire-truck' (red, man-made)..."
curl -s -X POST "$BASE_URL/vectors" \
    -H "Content-Type: application/json" \
    -d '{
        "id": "fire-truck",
        "vector": [0.8, 0.2, 0.1],
        "metadata": {"color": "red", "category": "vehicle"}
    }' | jq .
echo ""

# Step 5: List all vectors
echo "=== Step 5: Listing all stored vector IDs ==="
curl -s "$BASE_URL/vectors" | jq .
echo ""

# Step 6: Get a specific vector
echo "=== Step 6: Getting vector details for 'sunset' ==="
curl -s "$BASE_URL/vectors/sunset" | jq .
echo ""

# Step 7: Search — nearest to a "warm red" query
echo "=== Step 7: Searching for nearest neighbors ==="
echo "Query: [0.85, 0.15, 0.05] (warm red direction), k=3"
curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d '{
        "vector": [0.85, 0.15, 0.05],
        "k": 3
    }' | jq .
echo ""

# Step 8: Filtered search — only "red" things
echo "=== Step 8: Filtered search (color = red only) ==="
echo "Same query, but filtering to color=red..."
curl -s -X POST "$BASE_URL/search" \
    -H "Content-Type: application/json" \
    -d '{
        "vector": [0.85, 0.15, 0.05],
        "k": 3,
        "filter": {"op": "eq", "field": "color", "value": "red"}
    }' | jq .
echo ""

# Step 9: Batch insert
echo "=== Step 9: Batch insert (3 more vectors) ==="
curl -s -X POST "$BASE_URL/vectors/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "vectors": [
            {"id": "grass", "vector": [0.1, 0.8, 0.2], "metadata": {"color": "green", "category": "nature"}},
            {"id": "sky", "vector": [0.1, 0.3, 0.8], "metadata": {"color": "blue", "category": "nature"}},
            {"id": "taxi", "vector": [0.7, 0.7, 0.0], "metadata": {"color": "yellow", "category": "vehicle"}}
        ]
    }' | jq .
echo ""

# Step 10: Batch search
echo "=== Step 10: Batch search (2 queries at once) ==="
echo "Query 1: [0.9, 0.1, 0.0] (red direction)"
echo "Query 2: [0.0, 0.1, 0.9] (blue direction)"
curl -s -X POST "$BASE_URL/search/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "queries": [
            {"vector": [0.9, 0.1, 0.0], "k": 2},
            {"vector": [0.0, 0.1, 0.9], "k": 2}
        ]
    }' | jq .
echo ""

# Step 11: Batch search with filter
echo "=== Step 11: Batch search with filter (nature only) ==="
curl -s -X POST "$BASE_URL/search/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "queries": [
            {"vector": [0.9, 0.1, 0.0], "k": 2},
            {"vector": [0.0, 0.1, 0.9], "k": 2}
        ],
        "filter": {"op": "eq", "field": "category", "value": "nature"}
    }' | jq .
echo ""

# Step 12: Delete a vector
echo "=== Step 12: Deleting 'fire-truck' ==="
curl -s -X DELETE "$BASE_URL/vectors/fire-truck" | jq .
echo ""

# Step 13: Verify deletion
echo "=== Step 13: Verifying deletion (listing IDs) ==="
curl -s "$BASE_URL/vectors" | jq .
echo ""

# Step 14: Check metrics
echo "=== Step 14: Checking metrics ==="
curl -s "$BASE_URL/metrics" | jq .
echo ""

# Step 15: Final health check
echo "=== Step 15: Final health check ==="
curl -s "$BASE_URL/health" | jq .
echo ""

echo "============================================"
echo "  Demo complete!"
echo "============================================"
