#include <sys/utsname.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "core/types.h"
#include "segment/segment_manager.h"

// ── fvecs / ivecs parsing ───────────────────────────────────

static std::vector<float> load_fvecs(const std::string& path, int& dim, int& count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    dim = d;

    std::size_t vec_size = sizeof(int32_t) + d * sizeof(float);
    count = static_cast<int>(file_size / vec_size);

    std::vector<float> data(static_cast<std::size_t>(count) * d);
    in.seekg(0, std::ios::beg);

    for (int i = 0; i < count; i++) {
        int32_t dim_check = 0;
        in.read(reinterpret_cast<char*>(&dim_check), sizeof(dim_check));
        if (dim_check != d) throw std::runtime_error("fvecs: inconsistent dimension");
        in.read(reinterpret_cast<char*>(&data[static_cast<std::size_t>(i) * d]), d * sizeof(float));
    }
    return data;
}

static std::vector<std::vector<int32_t>> load_ivecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::vector<std::vector<int32_t>> result;
    while (in.peek() != std::ifstream::traits_type::eof()) {
        int32_t d = 0;
        in.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!in) break;
        std::vector<int32_t> vec(d);
        in.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int32_t));
        result.push_back(std::move(vec));
    }
    return result;
}

// ── Recall computation ──────────────────────────────────────

static double compute_recall(const std::vector<vexdb::QueryResult>& results,
                             const std::vector<int32_t>& gt, int at_k) {
    int hits = 0;
    std::set<int32_t> gt_set(gt.begin(), gt.begin() + std::min(at_k, static_cast<int>(gt.size())));
    int check = std::min(at_k, static_cast<int>(results.size()));
    for (int i = 0; i < check; i++) {
        if (gt_set.count(static_cast<int32_t>(results[i].user_id))) hits++;
    }
    return static_cast<double>(hits) / std::min(at_k, static_cast<int>(gt_set.size()));
}

// ── Machine info ────────────────────────────────────────────

struct MachineInfo {
    std::string cpu;
    std::string ram_gb;
    std::string os;
};

static MachineInfo get_machine_info() {
    MachineInfo info;

    // CPU: first "model name" line from /proc/cpuinfo
    if (std::ifstream f("/proc/cpuinfo"); f) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("model name", 0) == 0) {
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    info.cpu = line.substr(colon + 2);
                }
                break;
            }
        }
    }
    if (info.cpu.empty()) info.cpu = "unknown";

    // RAM: MemTotal from /proc/meminfo
    if (std::ifstream f("/proc/meminfo"); f) {
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("MemTotal", 0) == 0) {
                unsigned long long kb = 0;
                std::sscanf(line.c_str(), "MemTotal: %llu kB", &kb);
                char buf[16];
                std::snprintf(buf, sizeof(buf), "%.0f GB", kb / 1024.0 / 1024.0);
                info.ram_gb = buf;
                break;
            }
        }
    }
    if (info.ram_gb.empty()) info.ram_gb = "unknown";

    // OS: uname sysname + release
    struct utsname u {};
    if (uname(&u) == 0) {
        info.os = std::string(u.sysname) + " " + u.release;
    } else {
        info.os = "unknown";
    }

    return info;
}

// ── Config ──────────────────────────────────────────────────

struct BenchConfig {
    int m;
    int ef_construction;
    int ef_search;
};

struct BenchResult {
    BenchConfig config;
    double recall_at_1;
    double recall_at_10;
    double recall_at_100;
    int qps;
    double build_seconds;
};

// ── Main ────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string data_dir = "./data";
    if (argc > 1) data_dir = argv[1];

    auto machine = get_machine_info();
    std::cout << "Machine: " << machine.cpu << " | " << machine.ram_gb << " | " << machine.os
              << std::endl;

    std::cout << "Loading SIFT1M from " << data_dir << "..." << std::endl;

    int base_dim = 0, base_count = 0;
    auto base = load_fvecs(data_dir + "/sift_base.fvecs", base_dim, base_count);
    std::cout << "  Base: " << base_count << " vectors, " << base_dim << "d" << std::endl;

    int query_dim = 0, query_count = 0;
    auto queries = load_fvecs(data_dir + "/sift_query.fvecs", query_dim, query_count);
    std::cout << "  Queries: " << query_count << " vectors, " << query_dim << "d" << std::endl;

    auto gt = load_ivecs(data_dir + "/sift_groundtruth.ivecs");
    std::cout << "  Ground truth: " << gt.size() << " entries" << std::endl;

    if (base_dim != query_dim) {
        std::cerr << "ERROR: base dim " << base_dim << " != query dim " << query_dim << std::endl;
        return 1;
    }

    struct IndexConfig {
        int m;
        int ef_construction;
    };

    struct SearchConfig {
        int ef_search;
    };

    // Group by (M, ef_construction) to avoid redundant index builds.
    struct ConfigGroup {
        IndexConfig index;
        std::vector<SearchConfig> searches;
    };

    std::vector<ConfigGroup> groups = {
        {{16, 200}, {{64}, {128}, {256}}},
        {{32, 400}, {{128}, {256}}},
    };

    std::vector<BenchResult> results;

    for (const auto& group : groups) {
        std::cout << "\n=== Building: M=" << group.index.m
                  << " ef_construction=" << group.index.ef_construction << " ===" << std::endl;

        // Build via SegmentManager — capacity > base_count so no seal fires during build.
        vexdb::SegmentManager db(static_cast<vexdb::Dim>(base_dim), 1'100'000, "", group.index.m,
                                 group.index.ef_construction);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < base_count; i++) {
            if (i % 100000 == 0 && i > 0) {
                std::cout << "  Inserted " << i << "/" << base_count << std::endl;
            }
            db.insert(static_cast<vexdb::VectorId>(i),
                      {&base[static_cast<std::size_t>(i) * base_dim],
                       static_cast<std::size_t>(base_dim)});
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "  Build: " << std::fixed << std::setprecision(1) << build_s << "s ("
                  << static_cast<int>(base_count / build_s) << " vec/s)" << std::endl;

        // Sweep ef_search values on the same index.
        for (const auto& sc : group.searches) {
            std::cout << "\n--- ef_search=" << sc.ef_search << " ---" << std::endl;

            // Warmup: 100 queries to fill caches.
            int warmup = std::min(100, query_count);
            for (int q = 0; q < warmup; q++) {
                (void)db.search({&queries[static_cast<std::size_t>(q) * query_dim],
                                 static_cast<std::size_t>(query_dim)},
                                100, sc.ef_search);
            }

            // Pass 1: reine Suche — nur dieser Block wird für QPS gemessen.
            std::vector<std::vector<vexdb::QueryResult>> all_results(query_count);
            auto t2 = std::chrono::high_resolution_clock::now();
            for (int q = 0; q < query_count; q++) {
                all_results[q] = db.search({&queries[static_cast<std::size_t>(q) * query_dim],
                                            static_cast<std::size_t>(query_dim)},
                                           100, sc.ef_search);
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            // Pass 2: Recall-Berechnung außerhalb des Timers.
            double total_recall_1 = 0, total_recall_10 = 0, total_recall_100 = 0;
            for (int q = 0; q < query_count; q++) {
                total_recall_1 += compute_recall(all_results[q], gt[q], 1);
                total_recall_10 += compute_recall(all_results[q], gt[q], 10);
                total_recall_100 += compute_recall(all_results[q], gt[q], 100);
            }

            double search_s = std::chrono::duration<double>(t3 - t2).count();
            int qps = static_cast<int>(query_count / search_s);

            BenchResult r;
            r.config = {group.index.m, group.index.ef_construction, sc.ef_search};
            r.recall_at_1 = total_recall_1 / query_count;
            r.recall_at_10 = total_recall_10 / query_count;
            r.recall_at_100 = total_recall_100 / query_count;
            r.qps = qps;
            r.build_seconds = build_s;
            results.push_back(r);

            std::cout << "  Recall@1:   " << std::fixed << std::setprecision(4) << r.recall_at_1
                      << std::endl;
            std::cout << "  Recall@10:  " << r.recall_at_10 << std::endl;
            std::cout << "  Recall@100: " << r.recall_at_100 << std::endl;
            std::cout << "  QPS:        " << r.qps << std::endl;
        }
    }

    // Print markdown table.
    std::cout << "\n## Results\n\n";
    std::cout << "| M | ef_construction | ef_search | Recall@1 | Recall@10 | Recall@100 | QPS "
                 "| Build (s) |"
              << std::endl;
    std::cout << "|---|---|---|---|---|---|---|---|" << std::endl;
    for (const auto& r : results) {
        std::cout << "| " << r.config.m << " | " << r.config.ef_construction << " | "
                  << r.config.ef_search << " | " << std::fixed << std::setprecision(4)
                  << r.recall_at_1 << " | " << r.recall_at_10 << " | " << r.recall_at_100 << " | "
                  << r.qps << " | " << std::setprecision(1) << r.build_seconds << " |" << std::endl;
    }

    // Write results.md next to data_dir.
    auto results_path = std::filesystem::path(data_dir).parent_path() / "results.md";
    std::ofstream out(results_path);
    if (out) {
        out << "# SIFT1M Benchmark Results\n\n";
        out << "**Machine:** " << machine.cpu << " | " << machine.ram_gb << " | " << machine.os
            << "\n\n";
        out << "| M | ef_construction | ef_search | Recall@1 | Recall@10 | Recall@100 | QPS "
               "| Build (s) |\n";
        out << "|---|---|---|---|---|---|---|---|\n";
        for (const auto& r : results) {
            out << "| " << r.config.m << " | " << r.config.ef_construction << " | "
                << r.config.ef_search << " | " << std::fixed << std::setprecision(4)
                << r.recall_at_1 << " | " << r.recall_at_10 << " | " << r.recall_at_100 << " | "
                << r.qps << " | " << std::setprecision(1) << r.build_seconds << " |\n";
        }
    }

    // Check v0.1 gate.
    bool gate_passed = false;
    for (const auto& r : results) {
        if (r.recall_at_10 > 0.90) gate_passed = true;
    }
    std::cout << "\nv0.1 gate (Recall@10 > 0.90): " << (gate_passed ? "PASSED" : "FAILED")
              << std::endl;

    return gate_passed ? 0 : 1;
}
