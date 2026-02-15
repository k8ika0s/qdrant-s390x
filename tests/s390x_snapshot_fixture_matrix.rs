//! Snapshot fixture producer/consumer tests.
//!
//! These tests are `#[ignore]` and intended for cross-endian (LE<->BE) validation:
//!
//! - Run the producer on a little-endian machine to generate snapshot fixtures.
//! - Copy the produced fixture directory to a big-endian s390x host.
//! - Run the consumer on s390x to restore and validate the fixtures.
//!
//! By default, fixtures are written under `dev-docs/s390x-fixtures/<arch>_<endian>_<unix_ts>/`.
//! Override via `S390X_FIXTURES_DIR=/path/to/dir`.
//!
//! Note: Snapshot fixtures are stored gzipped (`*.snapshot.gz`) to avoid committing or transferring
//! large preallocated WAL/mmap files. The consumer inflates each fixture into a temp directory
//! before calling the Qdrant snapshot recovery API.

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

const ENV_FIXTURES_DIR: &str = "S390X_FIXTURES_DIR";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotFixtureManifest {
    format_version: u32,
    created_unix_utc: u64,
    arch: String,
    endian: String,
    fixtures: Vec<SnapshotFixtureEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotFixtureEntry {
    id: String,
    collection: String,
    snapshot_file: String,
}

#[test]
#[ignore]
fn s390x_snapshot_fixture_produce() {
    let out_dir = fixtures_dir_from_env_or_default();
    fs::create_dir_all(&out_dir).expect("create fixtures out dir");

    let tmp = TempDir::new().expect("create tempdir");

    // Keep snapshots in a shared path so we can copy them out after Qdrant exits.
    let snapshots_path = tmp.path().join("snapshots");
    let temp_path = tmp.path().join("tmp");
    fs::create_dir_all(&snapshots_path).expect("create snapshots dir");
    fs::create_dir_all(&temp_path).expect("create temp dir");

    let http_port = pick_unused_port();
    let grpc_port = pick_unused_port();
    let base_url = format!("http://127.0.0.1:{http_port}");
    let log_path = tmp.path().join("qdrant.log");

    // QEMU s390x runs can be significantly slower than native; keep timeouts generous
    // to avoid flaking the cross-endian producer/consumer gates.
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build http client");

    let storage = tmp.path().join("storage");
    fs::create_dir_all(&storage).expect("create storage dir");
    let mut qdrant = QdrantProc::spawn(
        &log_path,
        &storage,
        &snapshots_path,
        &temp_path,
        http_port,
        grpc_port,
    )
    .expect("spawn qdrant");
    wait_ready(&client, &base_url, &log_path);

    let mut fixtures = Vec::new();

    // Fixture 1: multi-vector + quantization + on-disk vectors (covers dense mmap + quantization).
    let multivec = "s390x_fixture_multivec";
    http_delete_collection_if_exists(&client, &base_url, multivec, &log_path);
    http_create_multivec_collection(&client, &base_url, multivec, &log_path);
    http_upsert_multivec_points(&client, &base_url, multivec, &log_path);
    http_search_multivec_and_assert(&client, &base_url, multivec, &log_path);
    let multivec_snapshot =
        http_create_collection_snapshot(&client, &base_url, multivec, &snapshots_path, &log_path);
    let multivec_snapshot_name = "multivec.snapshot.gz";
    gzip_fixture(&multivec_snapshot, &out_dir, multivec_snapshot_name);
    fixtures.push(SnapshotFixtureEntry {
        id: "multivec".to_string(),
        collection: multivec.to_string(),
        snapshot_file: multivec_snapshot_name.to_string(),
    });

    // Fixture 2: sparse vectors (covers inverted index persistence).
    let sparse = "s390x_fixture_sparse";
    http_delete_collection_if_exists(&client, &base_url, sparse, &log_path);
    http_create_sparse_collection(&client, &base_url, sparse, &log_path);
    http_upsert_sparse_points(&client, &base_url, sparse, &log_path);
    http_scroll_sparse_and_assert_sorted(&client, &base_url, sparse, &log_path);
    let sparse_snapshot =
        http_create_collection_snapshot(&client, &base_url, sparse, &snapshots_path, &log_path);
    let sparse_snapshot_name = "sparse.snapshot.gz";
    gzip_fixture(&sparse_snapshot, &out_dir, sparse_snapshot_name);
    fixtures.push(SnapshotFixtureEntry {
        id: "sparse".to_string(),
        collection: sparse.to_string(),
        snapshot_file: sparse_snapshot_name.to_string(),
    });

    qdrant.shutdown();

    let manifest = SnapshotFixtureManifest {
        format_version: 1,
        created_unix_utc: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs(),
        arch: env::consts::ARCH.to_string(),
        endian: if cfg!(target_endian = "big") {
            "big".to_string()
        } else {
            "little".to_string()
        },
        fixtures,
    };

    let manifest_path = out_dir.join(MANIFEST_FILE);
    let file = File::create(&manifest_path).expect("create manifest");
    serde_json::to_writer_pretty(file, &manifest).expect("write manifest");
}

#[test]
#[ignore]
fn s390x_snapshot_fixture_consume() {
    let in_dir = fixtures_dir_from_env();
    let manifest_path = in_dir.join(MANIFEST_FILE);

    let file = File::open(&manifest_path)
        .unwrap_or_else(|e| panic!("open manifest failed: {e} ({})", manifest_path.display()));
    let manifest: SnapshotFixtureManifest = serde_json::from_reader(file).unwrap_or_else(|e| {
        panic!(
            "parse manifest failed: {e} ({})",
            manifest_path.display()
        )
    });

    // QEMU s390x runs can be significantly slower than native; keep timeouts generous
    // to avoid flaking the cross-endian producer/consumer gates.
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build http client");

    for entry in &manifest.fixtures {
        let tmp = TempDir::new().expect("create tempdir");
        let snapshots_path = tmp.path().join("snapshots");
        let temp_path = tmp.path().join("tmp");
        fs::create_dir_all(&snapshots_path).expect("create snapshots dir");
        fs::create_dir_all(&temp_path).expect("create temp dir");

        let http_port = pick_unused_port();
        let grpc_port = pick_unused_port();
        let base_url = format!("http://127.0.0.1:{http_port}");
        let log_path = tmp.path().join("qdrant.log");

        let storage = tmp.path().join("storage");
        fs::create_dir_all(&storage).expect("create storage dir");
        let mut qdrant = QdrantProc::spawn(
            &log_path,
            &storage,
            &snapshots_path,
            &temp_path,
            http_port,
            grpc_port,
        )
        .expect("spawn qdrant");
        wait_ready(&client, &base_url, &log_path);

        let source_fixture = in_dir.join(&entry.snapshot_file);
        if !source_fixture.exists() {
            panic!(
                "missing fixture snapshot: {}",
                source_fixture.display()
            );
        }

        let snapshot_path = materialize_snapshot_fixture(&source_fixture, tmp.path());

        http_delete_collection_if_exists(&client, &base_url, &entry.collection, &log_path);
        http_recover_collection_from_snapshot(
            &client,
            &base_url,
            &entry.collection,
            &snapshot_path,
            &log_path,
        );

        match entry.id.as_str() {
            "multivec" => {
                http_collection_points_and_assert_at_least(
                    &client,
                    &base_url,
                    &entry.collection,
                    8,
                    &log_path,
                );
                http_search_multivec_and_assert(&client, &base_url, &entry.collection, &log_path);
            }
            "sparse" => {
                http_collection_points_and_assert_at_least(
                    &client,
                    &base_url,
                    &entry.collection,
                    3,
                    &log_path,
                );
                http_scroll_sparse_and_assert_sorted(&client, &base_url, &entry.collection, &log_path);
            }
            other => panic!("unknown fixture id: {other}"),
        }

        qdrant.shutdown();
    }
}

fn fixtures_dir_from_env() -> PathBuf {
    env::var_os(ENV_FIXTURES_DIR)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {ENV_FIXTURES_DIR} to a fixture directory produced by s390x_snapshot_fixture_produce"))
}

fn fixtures_dir_from_env_or_default() -> PathBuf {
    if let Some(dir) = env::var_os(ENV_FIXTURES_DIR) {
        return PathBuf::from(dir);
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_secs();
    let endian = if cfg!(target_endian = "big") {
        "big"
    } else {
        "little"
    };

    PathBuf::from(format!(
        "dev-docs/s390x-fixtures/{}_{}_{}",
        env::consts::ARCH,
        endian,
        ts
    ))
}

fn gzip_fixture(snapshot_path: &Path, out_dir: &Path, out_name: &str) {
    let out_path = out_dir.join(out_name);
    let input = File::open(snapshot_path).unwrap_or_else(|e| {
        panic!(
            "open snapshot for gzip failed: {e} ({})",
            snapshot_path.display()
        )
    });
    let output = File::create(&out_path)
        .unwrap_or_else(|e| panic!("create gz fixture failed: {e} ({})", out_path.display()));

    let mut encoder = GzEncoder::new(output, Compression::default());
    let mut input = std::io::BufReader::new(input);
    std::io::copy(&mut input, &mut encoder).expect("gzip copy");
    encoder.finish().expect("finish gzip");

    let size = fs::metadata(&out_path).expect("stat gz fixture").len();
    assert!(size > 0, "gz fixture is empty: {}", out_path.display());
}

fn materialize_snapshot_fixture(source_fixture: &Path, tmp_dir: &Path) -> PathBuf {
    let file_name = source_fixture
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| panic!("invalid fixture filename: {}", source_fixture.display()));

    if file_name.ends_with(".snapshot") {
        return source_fixture.to_path_buf();
    }

    if !file_name.ends_with(".snapshot.gz") {
        panic!("unsupported fixture type: {}", source_fixture.display());
    }

    let out_name = file_name.trim_end_matches(".gz");
    let out_path = tmp_dir.join(out_name);

    let input = File::open(source_fixture)
        .unwrap_or_else(|e| panic!("open gz fixture failed: {e} ({})", source_fixture.display()));
    let mut decoder = GzDecoder::new(std::io::BufReader::new(input));
    let output = File::create(&out_path)
        .unwrap_or_else(|e| panic!("create inflated fixture failed: {e} ({})", out_path.display()));
    let mut output = std::io::BufWriter::new(output);
    std::io::copy(&mut decoder, &mut output).expect("inflate gzip");

    let size = fs::metadata(&out_path)
        .expect("stat inflated fixture")
        .len();
    assert!(
        size > 0,
        "inflated fixture is empty: {}",
        out_path.display()
    );

    out_path
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

fn http_create_multivec_collection(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    // Small but meaningful multi-vector config:
    // - on-disk dense vectors -> chunked mmap vector storage
    // - scalar int8 quantization -> quantization persistence paths
    let body = json!({
        "vectors": {
            "image": {
                "size": 4,
                "distance": "Dot",
                "on_disk": true
            },
            "audio": {
                "size": 4,
                "distance": "Dot",
                "quantization_config": {
                    "scalar": { "type": "int8", "quantile": 0.6 }
                },
                "on_disk": true
            },
            "text": {
                "size": 8,
                "distance": "Cosine",
                "quantization_config": {
                    "scalar": { "type": "int8", "always_ram": true }
                },
                "on_disk": true
            }
        },
        "hnsw_config": { "m": 8, "ef_construct": 64 },
        "quantization": {
            "scalar": { "type": "int8", "quantile": 0.5 }
        },
        "optimizers_config": { "default_segment_number": 1 },
        "replication_factor": 1
    });

    let resp = client
        .put(format!("{base_url}/collections/{collection}"))
        .json(&body)
        .send()
        .unwrap_or_else(|e| {
            panic!(
                "create multivec collection request failed: {e}\n{}",
                tail_log(log_path)
            )
        });

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!(
            "create multivec collection failed: {status} {body}\n{}",
            tail_log(log_path)
        );
    }
}

fn http_upsert_multivec_points(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    // Keep this deterministic (no rng) so fixtures are reproducible.
    let points: Vec<_> = (1..=8)
        .map(|id| {
            let x = id as f32 / 10.0;
            json!({
                "id": id,
                "vector": {
                    "image": [x, 0.2, 0.3, 0.4],
                    "audio": [x, 0.2, 0.3, 0.4],
                    "text":  [x, 0.2, 0.3, 0.4, x, 0.2, 0.3, 0.4]
                },
                "payload": { "id": id }
            })
        })
        .collect();

    let body = json!({ "points": points });

    let resp = client
        .put(format!(
            "{base_url}/collections/{collection}/points?wait=true"
        ))
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("upsert multivec points request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("upsert multivec points failed: {status} {body}\n{}", tail_log(log_path));
    }
}

fn http_search_multivec_and_assert(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    let body = json!({
        "vector": { "name": "image", "vector": [0.2, 0.1, 0.9, 0.7] },
        "limit": 3
    });

    let resp = client
        .post(format!("{base_url}/collections/{collection}/points/search"))
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("multivec search request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("multivec search failed: {status} {body}\n{}", tail_log(log_path));
    }

    let v: serde_json::Value = resp
        .json()
        .unwrap_or_else(|e| panic!("parse multivec search response failed: {e}\n{}", tail_log(log_path)));
    let hits = v
        .get("result")
        .and_then(|r| r.as_array())
        .unwrap_or_else(|| panic!("search response missing result array: {v}\n{}", tail_log(log_path)));
    assert!(
        !hits.is_empty(),
        "expected at least one search hit\nresponse={v}\n{}",
        tail_log(log_path)
    );
}

fn http_create_sparse_collection(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    let body = json!({
        "sparse_vectors": {
            "text": {}
        },
        "optimizers_config": { "default_segment_number": 1 },
        "replication_factor": 1
    });

    let resp = client
        .put(format!("{base_url}/collections/{collection}"))
        .json(&body)
        .send()
        .unwrap_or_else(|e| {
            panic!(
                "create sparse collection request failed: {e}\n{}",
                tail_log(log_path)
            )
        });

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!(
            "create sparse collection failed: {status} {body}\n{}",
            tail_log(log_path)
        );
    }
}

fn http_upsert_sparse_points(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    let body = json!({
        "points": [
            { "id": 1, "vector": { "text": { "indices": [3, 2, 1], "values": [0.3, 0.2, 0.1] } } },
            { "id": 2, "vector": { "text": { "indices": [1, 3, 2], "values": [0.1, 0.3, 0.2] } } },
            { "id": 3, "vector": { "text": { "indices": [1, 2, 3], "values": [0.1, 0.2, 0.3] } } }
        ]
    });

    let resp = client
        .put(format!(
            "{base_url}/collections/{collection}/points?wait=true"
        ))
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("upsert sparse points request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("upsert sparse points failed: {status} {body}\n{}", tail_log(log_path));
    }
}

fn http_scroll_sparse_and_assert_sorted(client: &Client, base_url: &str, collection: &str, log_path: &Path) {
    let body = json!({ "limit": 10, "with_vector": true });

    let resp = client
        .post(format!("{base_url}/collections/{collection}/points/scroll"))
        .json(&body)
        .send()
        .unwrap_or_else(|e| panic!("sparse scroll request failed: {e}\n{}", tail_log(log_path)));

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        panic!("sparse scroll failed: {status} {body}\n{}", tail_log(log_path));
    }

    let v: serde_json::Value = resp
        .json()
        .unwrap_or_else(|e| panic!("parse sparse scroll response failed: {e}\n{}", tail_log(log_path)));

    let points = v
        .pointer("/result/points")
        .and_then(|p| p.as_array())
        .unwrap_or_else(|| panic!("scroll response missing result.points array: {v}\n{}", tail_log(log_path)));

    assert!(
        points.len() >= 3,
        "expected >= 3 points\nresponse={v}\n{}",
        tail_log(log_path)
    );

    for p in points {
        let indices = p
            .pointer("/vector/text/indices")
            .and_then(|x| x.as_array())
            .unwrap_or_else(|| panic!("missing vector.text.indices: {p}\n{}", tail_log(log_path)));
        let indices: Vec<u64> = indices.iter().map(|x| x.as_u64().unwrap()).collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(indices, sorted, "sparse indices must be sorted: {indices:?}");
    }
}

fn http_collection_points_and_assert_at_least(
    client: &Client,
    base_url: &str,
    collection: &str,
    min_points: u64,
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
        points >= min_points,
        "expected points_count >= {min_points}; got {points}\nresponse={v}\n{}",
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
