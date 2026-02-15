//! End-to-end snapshot create/restore smoke test for Qdrant.
//!
//! This test is `#[ignore]` to avoid impacting the default test runtime on all
//! architectures. It is intended to be run explicitly in s390x validation gates
//! (and can also be run on other targets).

use reqwest::blocking::Client;
use serde_json::json;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[test]
#[ignore]
fn s390x_snapshot_smoke_create_restore_roundtrip() {
    let tmp = TempDir::new().expect("create tempdir");

    // Keep snapshots in a shared path so we can restore into a fresh storage dir.
    let snapshots_path = tmp.path().join("snapshots");
    let temp_path = tmp.path().join("tmp");
    fs::create_dir_all(&snapshots_path).expect("create snapshots dir");
    fs::create_dir_all(&temp_path).expect("create temp dir");

    let http_port = pick_unused_port();
    let grpc_port = pick_unused_port();
    let base_url = format!("http://127.0.0.1:{http_port}");
    let log_path = tmp.path().join("qdrant.log");

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("build http client");

    let collection = "s390x_snapshot_smoke";

    // Boot #1: create data + create snapshot.
    let storage_a = tmp.path().join("storage_a");
    fs::create_dir_all(&storage_a).expect("create storage_a dir");
    let mut qdrant = QdrantProc::spawn(
        &log_path,
        &storage_a,
        &snapshots_path,
        &temp_path,
        http_port,
        grpc_port,
    )
    .expect("spawn qdrant");
    wait_ready(&client, &base_url, &log_path);

    http_delete_collection_if_exists(&client, &base_url, collection, &log_path);
    http_create_collection(&client, &base_url, collection, &log_path);
    http_upsert_points(&client, &base_url, collection, &log_path);
    http_search_and_assert(&client, &base_url, collection, &log_path);

    let snapshot_path = http_create_collection_snapshot(&client, &base_url, collection, &snapshots_path, &log_path);
    qdrant.shutdown();

    // Boot #2: restore snapshot into a fresh storage path (no existing collection).
    let storage_b = tmp.path().join("storage_b");
    fs::create_dir_all(&storage_b).expect("create storage_b dir");
    let mut qdrant = QdrantProc::spawn(
        &log_path,
        &storage_b,
        &snapshots_path,
        &temp_path,
        http_port,
        grpc_port,
    )
    .expect("spawn qdrant (restore)");
    wait_ready(&client, &base_url, &log_path);

    http_delete_collection_if_exists(&client, &base_url, collection, &log_path);
    http_recover_collection_from_snapshot(&client, &base_url, collection, &snapshot_path, &log_path);
    http_collection_info_and_assert(&client, &base_url, collection, &log_path);
    http_search_and_assert(&client, &base_url, collection, &log_path);

    // Cleanup after ourselves.
    http_delete_collection_if_exists(&client, &base_url, collection, &log_path);
    qdrant.shutdown();
}

fn pick_unused_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("read local addr")
        .port()
}

fn wait_ready(client: &Client, base_url: &str, log_path: &Path) {
    let start = Instant::now();
    loop {
        match client.get(format!("{base_url}/collections")).send() {
            Ok(resp) if resp.status().is_success() => return,
            _ => {
                if start.elapsed() > Duration::from_secs(30) {
                    panic!("qdrant did not become ready in time\n{}", tail_log(log_path));
                }
                thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

fn http_delete_collection_if_exists(
    client: &Client,
    base_url: &str,
    collection: &str,
    log_path: &Path,
) {
    let resp = client
        .delete(format!("{base_url}/collections/{collection}"))
        .send()
        .unwrap_or_else(|e| {
            panic!(
                "delete collection request failed: {e}\n{}",
                tail_log(log_path)
            )
        });

    // 200 OK (deleted) or 404 Not Found (already absent) are both acceptable.
    if !(resp.status().is_success() || resp.status().as_u16() == 404) {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("delete collection failed: {status} {body}\n{}", tail_log(log_path));
    }
}

fn http_create_collection(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    let body = json!({
        "vectors": { "size": 4, "distance": "Dot" },
        "optimizers_config": { "default_segment_number": 1 },
        "replication_factor": 1
    });

    let resp = client
        .put(format!("{base_url}/collections/{collection}"))
        .json(&body)
        .send()
        .unwrap_or_else(|e| {
            panic!(
                "create collection request failed: {e}\n{}",
                tail_log(log_path)
            )
        });

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("create collection failed: {status} {body}\n{}", tail_log(log_path));
    }
}

fn http_upsert_points(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    let body = json!({
        "points": [
            { "id": 1, "vector": [0.05, 0.61, 0.76, 0.74], "payload": { "city": "Berlin", "count": 1 } },
            { "id": 2, "vector": [0.19, 0.81, 0.75, 0.11], "payload": { "city": "London", "count": 2 } },
            { "id": 3, "vector": [0.36, 0.55, 0.47, 0.94], "payload": { "city": "Moscow", "count": 3 } }
        ]
    });

    let resp = client
        .put(format!(
            "{base_url}/collections/{collection}/points?wait=true"
        ))
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("upsert points request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("upsert points failed: {status} {body}\n{}", tail_log(log_path));
    }
}

fn http_search_and_assert(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    let body = json!({
        "vector": [0.2, 0.1, 0.9, 0.7],
        "top": 3
    });

    let resp = client
        .post(format!("{base_url}/collections/{collection}/points/search"))
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("search request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("search failed: {status} {body}\n{}", tail_log(log_path));
    }

    let v: serde_json::Value = resp
        .json()
        .unwrap_or_else(|e| panic!("parse search response failed: {e}\n{}", tail_log(log_path)));

    let hits = v.get("result").and_then(|r| r.as_array()).unwrap_or_else(|| {
        panic!(
            "search response missing result array: {v}\n{}",
            tail_log(log_path)
        )
    });

    assert!(
        !hits.is_empty(),
        "expected at least one search hit\nresponse={v}\n{}",
        tail_log(log_path)
    );
}

fn http_collection_info_and_assert(
    client: &Client,
    base_url: &str,
    collection: &str,
    log_path: &Path,
) {
    let resp = client
        .get(format!("{base_url}/collections/{collection}"))
        .send()
        .unwrap_or_else(|e| panic!("get collection request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("get collection failed: {status} {body}\n{}", tail_log(log_path));
    }

    let v: serde_json::Value = resp
        .json()
        .unwrap_or_else(|e| panic!("parse collection response failed: {e}\n{}", tail_log(log_path)));

    let points = v
        .pointer("/result/points_count")
        .and_then(|p| p.as_u64())
        .unwrap_or_else(|| {
            panic!(
                "collection response missing points_count: {v}\n{}",
                tail_log(log_path)
            )
        });

    assert!(
        points >= 3,
        "expected points_count >= 3 after restore; got {points}\nresponse={v}\n{}",
        tail_log(log_path)
    );
}

fn http_create_collection_snapshot(
    client: &Client,
    base_url: &str,
    collection: &str,
    snapshots_dir: &Path,
    log_path: &Path,
) -> PathBuf {
    let resp = client
        .post(format!(
            "{base_url}/collections/{collection}/snapshots?wait=true"
        ))
        .send()
        .unwrap_or_else(|e| panic!("create snapshot request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!(
            "create snapshot failed: {status} {body}\n{}",
            tail_log(log_path)
        );
    }

    let v: serde_json::Value = resp.json().unwrap_or_else(|e| {
        panic!(
            "parse create snapshot response failed: {e}\n{}",
            tail_log(log_path)
        )
    });

    let name = v
        .pointer("/result/name")
        .and_then(|n| n.as_str())
        .unwrap_or_else(|| panic!("snapshot response missing result.name: {v}\n{}", tail_log(log_path)));

    // Collection snapshots live under `<snapshots_path>/<collection>/<snapshot_name>`.
    let snapshot_path = snapshots_dir.join(collection).join(name);

    // Snapshot creation can involve background fsync/rename on some platforms; wait briefly.
    let start = Instant::now();
    while !snapshot_path.exists() {
        if start.elapsed() > Duration::from_secs(30) {
            panic!(
                "snapshot file did not appear: {}\nresponse={v}\n{}",
                snapshot_path.display(),
                tail_log(log_path)
            );
        }
        thread::sleep(Duration::from_millis(200));
    }

    snapshot_path
}

fn http_recover_collection_from_snapshot(
    client: &Client,
    base_url: &str,
    collection: &str,
    snapshot_path: &Path,
    log_path: &Path,
) {
    let location = format!("file://{}", snapshot_path.display());
    let body = json!({ "location": location });

    let resp = client
        .put(format!(
            "{base_url}/collections/{collection}/snapshots/recover?wait=true"
        ))
        .json(&body)
        .send()
        .unwrap_or_else(|e| {
            panic!(
                "recover snapshot request failed: {e}\n{}",
                tail_log(log_path)
            )
        });

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!(
            "recover snapshot failed: {status} {body}\n{}",
            tail_log(log_path)
        );
    }
}

struct QdrantProc {
    child: Child,
    is_shutdown: bool,
}

impl QdrantProc {
    fn spawn(
        log_path: &Path,
        storage_path: &Path,
        snapshots_path: &Path,
        temp_path: &Path,
        http_port: u16,
        grpc_port: u16,
    ) -> std::io::Result<Self> {
        let log = File::create(log_path)?;
        let log_err = log.try_clone()?;

        let mut cmd = Command::new(env!("CARGO_BIN_EXE_qdrant"));
        cmd.env("QDRANT__SERVICE__HOST", "127.0.0.1")
            .env("QDRANT__SERVICE__HTTP_PORT", http_port.to_string())
            .env("QDRANT__SERVICE__GRPC_PORT", grpc_port.to_string())
            .env("QDRANT__STORAGE__STORAGE_PATH", storage_path)
            .env("QDRANT__STORAGE__SNAPSHOTS_PATH", snapshots_path)
            .env("QDRANT__STORAGE__TEMP_PATH", temp_path)
            .env("QDRANT__TELEMETRY_DISABLED", "true")
            .env("RUST_LOG", "warn")
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));

        let child = cmd.spawn()?;
        Ok(Self {
            child,
            is_shutdown: false,
        })
    }

    fn shutdown(&mut self) {
        if self.is_shutdown {
            return;
        }

        // Prefer a graceful shutdown so storage state is cleanly persisted.
        #[cfg(unix)]
        {
            // Avoid adding extra crate features just for signal support in this smoke test.
            let _ = Command::new("kill")
                .arg("-2")
                .arg(self.child.id().to_string())
                .status();
        }

        let start = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => {
                    self.is_shutdown = true;
                    return;
                }
                Ok(None) => {
                    if start.elapsed() > Duration::from_secs(10) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                Err(_) => break,
            }
        }

        let _ = self.child.kill();
        let _ = self.child.wait();
        self.is_shutdown = true;
    }
}

impl Drop for QdrantProc {
    fn drop(&mut self) {
        if !self.is_shutdown {
            // Best-effort cleanup; never panic in Drop.
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn tail_log(path: &Path) -> String {
    // Best-effort tail; avoid panicking while building an error message.
    const MAX_BYTES: u64 = 16 * 1024;

    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };

    let len = match file.metadata() {
        Ok(m) => m.len(),
        Err(_) => return String::new(),
    };

    let start = len.saturating_sub(MAX_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return String::new();
    }

    let mut buf = Vec::new();
    if file.read_to_end(&mut buf).is_err() {
        return String::new();
    }

    let s = String::from_utf8_lossy(&buf);
    if s.is_empty() {
        String::new()
    } else {
        format!("--- qdrant log (tail) ---\n{s}")
    }
}
