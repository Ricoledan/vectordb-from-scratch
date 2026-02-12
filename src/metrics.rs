//! Observability metrics: query latency, insert/delete throughput, index stats.

use std::time::Duration;

/// Collects runtime metrics for the vector database.
#[derive(Debug)]
pub struct MetricsCollector {
    query_latencies_us: Vec<f64>,
    total_queries: u64,
    total_inserts: u64,
    total_deletes: u64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            query_latencies_us: Vec::new(),
            total_queries: 0,
            total_inserts: 0,
            total_deletes: 0,
        }
    }

    /// Record a query with its duration.
    pub fn record_query(&mut self, duration: Duration) {
        self.total_queries += 1;
        self.query_latencies_us.push(duration.as_micros() as f64);
    }

    /// Record an insert operation.
    pub fn record_insert(&mut self) {
        self.total_inserts += 1;
    }

    /// Record a delete operation.
    pub fn record_delete(&mut self) {
        self.total_deletes += 1;
    }

    pub fn total_queries(&self) -> u64 {
        self.total_queries
    }

    pub fn total_inserts(&self) -> u64 {
        self.total_inserts
    }

    pub fn total_deletes(&self) -> u64 {
        self.total_deletes
    }

    /// Average query latency in microseconds.
    pub fn avg_query_latency_us(&self) -> f64 {
        if self.query_latencies_us.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.query_latencies_us.iter().sum();
        sum / self.query_latencies_us.len() as f64
    }

    /// Get a percentile of query latency (e.g., 50.0, 95.0, 99.0).
    pub fn percentile_query_latency_us(&self, percentile: f64) -> f64 {
        if self.query_latencies_us.is_empty() {
            return 0.0;
        }

        let mut sorted = self.query_latencies_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_basic() {
        let mut m = MetricsCollector::new();
        m.record_insert();
        m.record_insert();
        m.record_delete();

        assert_eq!(m.total_inserts(), 2);
        assert_eq!(m.total_deletes(), 1);
        assert_eq!(m.total_queries(), 0);
    }

    #[test]
    fn test_metrics_latency() {
        let mut m = MetricsCollector::new();
        m.record_query(Duration::from_micros(100));
        m.record_query(Duration::from_micros(200));
        m.record_query(Duration::from_micros(300));

        assert_eq!(m.total_queries(), 3);
        assert!((m.avg_query_latency_us() - 200.0).abs() < 1.0);
        assert!((m.percentile_query_latency_us(50.0) - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_metrics_empty() {
        let m = MetricsCollector::new();
        assert_eq!(m.avg_query_latency_us(), 0.0);
        assert_eq!(m.percentile_query_latency_us(99.0), 0.0);
    }
}
