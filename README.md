# vexdb

A segmented vector database for Approximate Nearest Neighbor search, written in C++20. Uses AVX2 SIMD distance kernels, HNSW graph indexing, and memory-mapped storage.

Vectors are written to an active in-memory segment, sealed to disk when full, and served as read-only mmap-backed segments. Search fans out across all segments and merges results.

## Architecture

```mermaid
graph LR
    A[core] --> B[store] --> C[index] --> D[segment] --> E[persistence]
```

| Layer | Contents |
|---|---|
| **core** | types, L2 distance (AVX2 + scalar, compile-time dispatch), SQ8 quantization |
| **store** | VectorStore concept, InMemoryStore, MmapStore |
| **index** | HnswIndex\<Store\>, FlatIndex\<Store\> |
| **segment** | ActiveSegment, SealedSegment, SegmentManager |
| **persistence** | Serializer, Loader, VEX0/HNSW/IDS binary formats |

Templates live in `store/` and `index/`. Everything from `segment/` up exposes only concrete types.

See [docs/PRD.md](docs/PRD.md) for the full design rationale.

## Code style

Code follows STL/snake_case naming convention: types in `PascalCase`, functions and variables in `snake_case`, namespace `vexdb` in lowercase.

## Build

Requires CMake 3.20+, a C++20 compiler (GCC 14+ or Clang 18+).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run tests and benchmarks:

```bash
ctest --test-dir build --output-on-failure
./build/benchmarks/vexdb_bench
```

### SIFT1M benchmark (optional)

Requires the SIFT1M dataset (~160 MB download).

```bash
./benchmarks/sift1m/download.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVEXDB_BUILD_SIFT1M=ON
cmake --build build
cd benchmarks/sift1m && ../../build/benchmarks/sift1m/vexdb_sift1m
```

Results are written to `benchmarks/sift1m/results.md`.

### Python bindings (optional)

Requires Python 3.8+ and NumPy.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVEXDB_BUILD_PYTHON=ON
cmake --build build
PYTHONPATH=build/python python3 -c "import vexdb; print('ok')"
```

## Usage

### C++

```cpp
#include <vector>
#include "segment/segment_manager.h"

// In-memory only
vexdb::SegmentManager db(/*dim=*/768, /*segment_capacity=*/1000000);

// Insert
std::vector<float> vec(768, 0.0f);
db.insert(/*user_id=*/42, vec.data());

// Search
std::vector<float> query(768, 1.0f);
auto results = db.search(query.data(), /*k=*/10);
for (const auto& r : results) {
    // r.user_id, r.distance
}

// With persistence
vexdb::SegmentManager db2(768, 1000000, "path/to/db");
db2.insert(42, vec.data());
db2.save();
auto loaded = vexdb::SegmentManager::load("path/to/db");
```

### Python

```python
import numpy as np
import vexdb

db = vexdb.Database(dimensions=768, segment_capacity=1_000_000, path="path/to/db")

db.insert(user_id=42, vector=np.random.randn(768).astype(np.float32))

results = db.search(query=np.random.randn(768).astype(np.float32), k=10)
for user_id, distance in results:
    print(f"  {user_id}: {distance:.4f}")

db.save()
db2 = vexdb.Database.load("path/to/db")
```

## Benchmarks

Release build, single-threaded. Selected results from local runs:

| | Time |
|---|---|
| L2 distance (scalar, 128d) | 43 ns |
| L2 distance (AVX2, 128d) | 9 ns |
| L2 distance (AVX2, 768d) | 80 ns |
| FlatIndex search (10K, 128d) | 163 μs |
| HNSW search (10K, 128d) | 39 μs |
| HNSW search (100K, 128d) | 145 μs |

HNSW is 4.2x faster than brute-force at 10K vectors. At 100K, HNSW search time grows sub-linearly (39 μs → 145 μs for 10x more vectors).

### SIFT1M results

1M vectors, 128d float32, single-threaded, via `SegmentManager` (capacity 1.1M, no seal during build).

**Machine:** 12th Gen Intel(R) Core(TM) i7-1260P | 12 GB RAM | Linux 5.15.153.1-microsoft-standard-WSL2

| M | ef_construction | ef_search | Recall@1 | Recall@10 | Recall@100 | QPS | Build (s) |
|---|---|---|---|---|---|---|---|
| 16 | 200 | 64 | 0.9902 | 0.9903 | 0.9478 | 3503 | 603.7 |
| 16 | 200 | 128 | 0.9919 | 0.9941 | 0.9664 | 2810 | 603.7 |
| 16 | 200 | 256 | 0.9939 | 0.9986 | 0.9923 | 1523 | 603.7 |
| 32 | 400 | 128 | 0.9937 | 0.9985 | 0.9911 | 1605 | 1868.3 |
| 32 | 400 | 256 | 0.9940 | 0.9993 | 0.9986 | 944 | 1868.3 |

v0.1 gate (Recall@10 > 0.90): **PASSED** — alle 5 Configs erfüllen das Kriterium.

## Project status

v0.1 — MVP. Single-node, single-threaded. See [milestones](https://github.com/mariorch22/vexdb/milestones) for the roadmap.

## License

MIT
