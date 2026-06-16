# vextor — Product Requirements Document

## Overview

vextor is a segmented vector database for Approximate Nearest Neighbor (ANN) search, written in C++20 with AVX2 SIMD distance kernels.

Vectors are written to an active in-memory segment. When it fills up, it gets sealed to disk and becomes immutable. Sealed segments can be loaded back via mmap (zero-copy) or into RAM. Search fans out across all segments and merges results.

## Problem

I wanted to understand how vector databases actually work — not the API, but the internals. How does HNSW behave when you have more vectors than fit in RAM? What happens at segment boundaries during search? Why does hnswlib not have persistence built in, and what would it take to add it cleanly? How much does SQ8 actually cost you in recall, and where does the asymmetric trick recover it?

Reading papers and source code gets you part of the way. Building the whole stack from distance kernels to segment lifecycle to on-disk persistence gets you the rest.

## Goals

- Provide a single-node vector database with HNSW-based ANN search, segment lifecycle management, and on-disk persistence.
- Achieve competitive recall and latency on standard benchmarks (SIFT1M, GloVe).
- Support both in-memory and memory-mapped storage backends transparently.
- Expose a minimal, ergonomic API via C++ and Python (nanobind).
- Maintain a layered architecture that does not preclude future GPU backends or multi-node distribution.

## Non-goals

- **Distributed coordination.** The segment architecture would support sharding across nodes, but building the coordination layer (consensus, replication, rebalancing) is a different project entirely. Out of scope.
- **Streaming ingestion.** Batch inserts. The seal-on-full design assumes you're not getting 10K concurrent writes.
- **Multi-tenancy.** This is a library, not a service.
- **Hybrid search.** No BM25, no filtering, no metadata. Vector similarity only — adding a keyword index would double the scope for a feature that's orthogonal to what I'm trying to learn here.

## Target user

Primary audience: engineering teams evaluating the author's systems-level skills — the codebase, benchmarks, and architectural decisions are the deliverable.

Secondary audience: developers who want an embeddable vector search library for prototyping or small-scale use. The API is designed to be usable, but production hardening (fuzz testing, adversarial inputs, long-running stability) is a stretch goal beyond v0.2, not a launch requirement.

## Core features

### v0.1 — Single-node MVP

- **Segment lifecycle**: ActiveSegment (mutable, in-memory) → seal() → SealedSegment (immutable, read-only). When an insert hits the capacity limit, the SegmentManager seals the active segment synchronously (persist to disk, convert to SealedSegment), then creates a new empty ActiveSegment. The insert that triggered the seal is written to the new segment. The caller blocks during seal — this is acceptable for batch-insert workloads and avoids the complexity of background sealing with concurrent access.
- **HNSW index**: multi-layer graph with bidirectional edges, greedy search, neighbor pruning.
- **Storage backends**: InMemoryStore (std::vector\<float\>) and MmapStore (mmap, zero-copy). Unified via C++20 VectorStore concept.
- **Persistence**: Binary formats for vectors (VEX0), graph (HNSW), and ID mapping. JSON metadata for segment registry.
- **SegmentManager**: Holds one ActiveSegment and N SealedSegments. Search fans out to all segments and merges global top-k.
- **Distance functions**: L2 with AVX2 SIMD (8-wide float) and scalar fallback. Compile-time dispatch.
- **ID mapping**: Bidirectional translation between user-provided uint64 IDs and internal offsets.
- **Python bindings**: nanobind wrapper exposing insert, search, save, load.

### v0.2 — Performance

- **Parallel search**: Thread pool for concurrent search across sealed segments.
- **Pluggable distance functions**: Template parameter on `HnswIndex` and `FlatIndex` for L2, cosine, and inner product. Required foundation for SQ8 integration.
- **Pluggable index type**: Template `Segment` over the index type (`HnswIndex` / `FlatIndex`) so users can choose brute-force for small segments where HNSW overhead isn't worth it. Done in the same template-parameter pass as `DistanceFunc` — `Segment<IndexType, DistanceFunc>` — to avoid two separate refactors.
- **SQ8 integration**: Implement a quantized store type and asymmetric distance dispatch (float32 query vs. uint8 database vectors). Requires pluggable distance functions above.
- **Product Quantization (PQ)**: Compressed vector representation with asymmetric distance computation for reduced memory footprint and faster scan.
- **Benchmarking suite**: Automated recall/latency/throughput measurement on SIFT1M with CI integration.

### v0.3 — GPU backend

- **CUDA flat-index kernel**: Batch L2 distance computation on GPU for segments that fit in VRAM.
- **GPU store**: VRAM-resident vector storage with host↔device transfer management.
- **Baseline benchmark**: Compare GPU flat-scan vs. CPU HNSW on the same dataset at various scales to determine where the crossover point lies.

## API contract

### C++

```cpp
#include <vextor/vextor.h>

// Create a database. The path is optional — without it, the database
// is in-memory only and save() is unavailable.
vextor::Database db(dimensions, segment_capacity, "path/to/db");

// Insert vectors (std::span<const float>; batch insert is Python-only for now)
db.insert(user_id, vector_data);

// Search
auto results = db.search(query_vector, k);
// results: std::vector<QueryResult> → {user_id, distance}

// Persistence — writes to the path given at construction
db.save();
auto db2 = vextor::Database::load("path/to/db");
```

### Python

```python
import vextor

db = vextor.Database(dimensions=768, segment_capacity=1_000_000, path="path/to/db")

db.insert(user_id=42, vector=embedding)            # 1D float32 ndarray
db.insert_batch(user_ids=ids, vectors=embeddings)  # uint64 ndarray + 2D float32 ndarray

results = db.search(query=query_embedding, k=10)
# [(user_id, distance), ...]

db.save()  # writes to the path given at construction
db = vextor.Database.load("path/to/db")
```

## Performance budgets

All benchmarks on SIFT1M (1M vectors, 128 dimensions), single-threaded unless noted, measured on a consumer desktop CPU. Measured values: i7-1260P (WSL2), M=16, ef_construction=200, ef_search=128 — see README for the full sweep.

| Metric | v0.1 target | v0.1.0 measured | v0.2 target |
|---|---|---|---|
| Recall@10 | > 90% | 99.4% | > 95% (with PQ) |
| Search latency (single query) | < 5 ms | ~0.36 ms | < 2 ms |
| Search throughput (batch, parallel) | — | 2810 QPS (single-threaded) | > 1000 QPS |
| Insert throughput | > 5K vec/s | ~1.7K vec/s | > 10K vec/s |
| Memory per vector (128d, float32) | ~700 bytes | — | ~200 bytes (PQ) |

Insert throughput misses the v0.1 target: the 5K vec/s budget was set before tuning, and the shipped configuration spends its budget on recall instead (ef_construction=200 yields 99.4% Recall@10 against a 90% gate). Closing the gap is explicit v0.2 work (bulk persistence I/O, HNSW insert hot-path). Conversely, single-threaded search throughput already exceeds the v0.2 parallel target — that target will be revised upward once parallel search lands.

Memory breakdown (128d, M=16): HNSW graph ~160 bytes + ID mapping 8 bytes is fixed overhead per vector regardless of quantization. Vector payload: float32 = 512 bytes, PQ-16 = 16 bytes.

## Architecture

Five-layer design with strict top-down dependency flow:

```
core/          → types, distance functions (no dependencies)
store/         → VectorStore concept, InMemoryStore, MmapStore
index/         → HnswIndex<Store>, FlatIndex<Store> (templated over store)
segment/       → ActiveSegment, SealedSegment, SegmentManager (concrete types)
persistence/   → Serializer, Loader, VEX0 format definitions
```

Templates live in store/ and index/. segment/ and persistence/ expose only concrete types — the template boundary is sealed at the segment layer. In v0.2 `Segment` is templatized over both `IndexType` (`HnswIndex` / `FlatIndex`) and `DistanceFunc` in a single pass, keeping the two concerns orthogonal without layering separate refactors.

## Concurrency model

v0.1 is single-threaded. No concurrent inserts, no concurrent insert + search. This is a deliberate simplification — the segment lifecycle and persistence layer are designed first, concurrency layered on top in v0.2.

v0.2 introduces parallel search across sealed segments. The guarantees:

- **SealedSegments are immutable and lock-free.** Multiple threads can search them concurrently with zero synchronization. This is the main payoff of the Active/Sealed split. Caveat: the current `HnswIndex::search` keeps shared mutable scratch state (generation-based visited tracking), so this guarantee requires per-thread search scratch first — tracked in [#51](https://github.com/mariorch22/vextor/issues/51) as a prerequisite for parallel search.
- **ActiveSegment requires a read-write lock.** Concurrent searches are allowed (shared lock), inserts take an exclusive lock. Insert + search on the active segment are serialized against each other but not against sealed segment searches.
- **Seal is a stop-the-world operation.** During seal, the active segment is exclusively locked — inserts and searches on it block until the new ActiveSegment is ready. Searches on sealed segments are unaffected.

## Crash recovery

No write-ahead log. The design accepts data loss of the current ActiveSegment on crash — only sealed and persisted segments survive. This is an explicit tradeoff: WAL complexity is not justified for a batch-insert workload where the source data (embeddings from a model) is reproducible.

Persistence is atomic at the segment level from the loader's perspective: seal() writes the three segment files (vectors.bin, hnsw.bin, ids.bin) and then updates segments.json — the registry that decides what load() reads. The registry is replaced via temp file + rename, so a crash mid-write should not expose torn JSON. Without fsync, power-loss durability still depends on the filesystem; the implementation aims to reduce the data-loss window to the current ActiveSegment, but does not claim full crash-durable commits. A *referenced* segment with missing or truncated files fails load() with an explicit error — no partial state is loaded.

## Correctness and testing

- **Unit tests (GoogleTest)**: Every layer has isolated tests. Distance functions tested against naive implementations, HNSW invariants (bidirectional edges, layer assignment distribution) verified after bulk inserts, serializer round-trip tested (write → load → binary-compare).
- **Benchmarks (Google Benchmark)**: Microbenchmarks for distance kernels, HNSW search, and insert throughput. Cross-commit regression tracking is planned as part of the v0.2 benchmarking suite.
- **Sanitizers in CI**: ASan and UBSan on every Debug build (GCC 14 and Clang 18 matrix), clang-format enforced. MSan for the serialization path is planned, not yet wired up.
- **Recall regression in CI (planned, v0.2)**: On every PR, run search on a small fixed dataset (SIFT10K) and assert Recall@10 ≥ threshold, preventing silent algorithmic regressions. Until then, recall is validated manually via the SIFT1M benchmark.
- **Fuzz testing (stretch goal, v0.2)**: libFuzzer on the persistence loader to catch malformed-file crashes.

## Open questions

- **PQ config**: 8 or 16 subquantizers? 256 centroids per sub? No idea yet — need to bench on SIFT1M and see where recall falls off.
- **Thread pool**: Custom pool vs. BS::thread_pool vs. waiting for std::execution to be usable. Leaning toward BS::thread_pool for now.
- **GPU memory**: Pre-allocate per segment or shared pool with eviction? Depends on how the v0.3 benchmarks shake out. At 768d float32, a single 1M-vector segment is ~3 GB — that's most of a consumer GPU's VRAM already.
- **GPU vs. CPU crossover**: Where exactly does GPU flat-scan stop beating CPU HNSW? The "adaptive backend" idea sounds clean but the decision boundary depends on dimensionality, dataset size, and GPU model. Need real numbers before designing anything.
- **Deletion**: Append-only for now. The path is probably tombstone bitmaps checked during search + background compaction that rewrites segments without dead vectors. But compaction scheduling and the interaction with mmap'd segments is non-trivial, so this is firmly post-v0.2.
- **Quantization-aware HNSW**: Build the graph with full-precision vectors but search with PQ? Or construct on compressed representations too? Literature suggests full-precision construction + compressed search, but that means storing both representations during build. Need to think about the memory implications.

## Milestones

### v0.1.0 — MVP

Scope: HNSW, dual-store, segment lifecycle, persistence, Python bindings.

Done when:
- All GoogleTest tests green, ASan/UBSan clean
- Recall@10 > 90% on SIFT1M demonstrated (logged, not just asserted)
- Insert 1M vectors, save, load, search — full round-trip works
- `pip install` from built wheel works and exposes insert/search/save/load
- README documents build instructions, architecture, and a usage example

### v0.2.0 — Performance

Scope: Parallel search, Product Quantization, benchmarking suite.

Done when:
- Search latency < 2ms single-query on SIFT1M
- Recall@10 > 95% with PQ on SIFT1M
- Benchmark suite runs in CI and produces recall/latency/throughput numbers on every tagged release
- Parallel search across sealed segments demonstrably faster than sequential (benchmark proof)

### v0.3.0 — GPU

Scope: CUDA flat-index kernel, GPU store, CPU vs. GPU comparison.

Done when:
- CUDA kernel passes the same correctness tests as CPU path
- Benchmark comparison: GPU flat-scan vs. CPU HNSW at 100K, 1M, 10M vectors with published results
- Clear documentation of when GPU wins and when it doesn't
