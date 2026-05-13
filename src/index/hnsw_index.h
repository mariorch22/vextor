#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/distance.h"
#include "core/search_result.h"
#include "core/types.h"
#include "store/concept.h"

namespace vexdb {

// HNSW graph data: layer assignments and adjacency lists.
struct HnswGraph {
    int max_level = 0;
    int m = 0;
    Offset entry_point = 0;
    bool empty = true;

    std::vector<int> levels;        // level assignment per node
    std::vector<Offset> layer0;     // flat adjacency list for layer 0
    std::vector<int> layer0_count;  // actual neighbor count per node at layer 0
    std::vector<std::vector<std::vector<Offset>>> upper;  // upper layer adjacency lists

    std::size_t size() const { return levels.size(); }
    int max_m0() const { return 2 * m; }
    int layer0_stride() const { return max_m0() + 1; }  // +1 for temporary overflow during pruning
};

template <VectorStore Store>
class HnswIndex {
   public:
    HnswIndex(const Store& store, int m, int ef_construction = 200)
        : store_(store), ef_construction_(ef_construction) {
        if (m <= 1) throw std::invalid_argument("HnswIndex: m must be > 1");
        if (ef_construction <= 0)
            throw std::invalid_argument("HnswIndex: ef_construction must be > 0");
        graph_.m = m;
    }

    // Access to graph structure (for persistence and testing).
    const HnswGraph& graph() const { return graph_; }
    HnswGraph& graph() { return graph_; }

    // ── Insert ───────────────────────────────────────────────

    void insert() {
        if (graph_.size() >= store_.size()) {
            throw std::out_of_range("HnswIndex: no vector available for insert");
        }

        Offset vector_id = static_cast<Offset>(graph_.size());
        float u = std::max(level_dist_(rng_), std::numeric_limits<float>::min());
        int level = static_cast<int>(-std::log(u) * (1.0 / std::log(graph_.m)));

        if (graph_.empty) {
            graph_.entry_point = vector_id;
            graph_.max_level = level;
            graph_.empty = false;
            add_node(level);
            return;
        }

        add_node(level);
        Dim dim = store_.dimensions();
        const float* query = store_.get_vector(vector_id);

        Offset start = graph_.entry_point;
        for (int layer = graph_.max_level; layer > level; layer--) {
            start = greedy_search(query, layer, start, 1)[0];
        }

        for (int layer = std::min(level, graph_.max_level); layer >= 0; layer--) {
            int mmax = max_connections(layer);

            auto candidates = greedy_search(query, layer, start, ef_construction_);
            start = candidates[0];

            auto selected = select_neighbors_heuristic(query, candidates, mmax);
            set_neighbors(vector_id, layer, selected);

            for (Offset nb : selected) {
                if (has_neighbor(nb, layer, vector_id)) continue;
                add_neighbor(nb, layer, vector_id);

                if (neighbor_count(nb, layer) > mmax) {
                    const float* nb_vec = store_.get_vector(nb);
                    auto nb_nbs = get_neighbors(nb, layer);

                    std::vector<std::pair<float, Offset>> scored;
                    scored.reserve(nb_nbs.size());
                    for (Offset n : nb_nbs) {
                        scored.push_back({l2_distance(nb_vec, store_.get_vector(n), dim), n});
                    }
                    std::sort(scored.begin(), scored.end());

                    std::vector<Offset> sorted_ids;
                    sorted_ids.reserve(scored.size());
                    for (auto& [d, id] : scored) {
                        sorted_ids.push_back(id);
                    }

                    std::vector<Offset> old_nbs(nb_nbs.begin(), nb_nbs.end());
                    auto new_nbs = select_neighbors_heuristic(nb_vec, sorted_ids, mmax);
                    set_neighbors(nb, layer, new_nbs);

                    // Remove back-edges for pruned nodes.
                    std::set<Offset> new_set(new_nbs.begin(), new_nbs.end());
                    for (Offset old_nb : old_nbs) {
                        if (!new_set.count(old_nb)) {
                            remove_neighbor(old_nb, layer, nb);
                        }
                    }
                    // If we got pruned from nb, remove our forward edge too.
                    if (!new_set.count(vector_id)) {
                        remove_neighbor(vector_id, layer, nb);
                    }
                }
            }
        }

        if (level > graph_.max_level) {
            graph_.entry_point = vector_id;
            graph_.max_level = level;
        }
    }

    // ── Search ───────────────────────────────────────────────

    std::vector<SearchResult> search(const float* query, std::size_t k, int ef_search = 64) const {
        if (graph_.empty || k == 0) return {};

        ef_search = std::max(static_cast<int>(k), ef_search);

        Offset current = graph_.entry_point;
        for (int layer = graph_.max_level; layer >= 1; layer--) {
            current = greedy_search(query, layer, current, 1)[0];
        }

        auto candidates = greedy_search(query, 0, current, ef_search);

        Dim dim = store_.dimensions();
        std::vector<SearchResult> results;
        results.reserve(candidates.size());
        for (Offset id : candidates) {
            results.push_back(
                {.offset = id, .distance = l2_distance(query, store_.get_vector(id), dim)});
        }

        std::sort(results.begin(), results.end());
        if (results.size() > k) results.resize(k);
        return results;
    }

   private:
    const Store& store_;
    HnswGraph graph_;
    int ef_construction_;
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> level_dist_{0.0f, 1.0f};

    // Generation-based visited tracking: avoids clearing the array each search.
    // Not thread-safe: shared scratch state for single-threaded search.
    mutable std::vector<uint32_t> visited_gen_;
    mutable uint32_t current_gen_ = 0;

    // ── Graph structure helpers ──────────────────────────────

    int max_m0() const { return graph_.max_m0(); }
    int layer0_stride() const { return graph_.layer0_stride(); }
    int max_connections(int layer) const { return layer == 0 ? max_m0() : graph_.m; }

    void add_node(int level) {
        graph_.levels.push_back(level);
        graph_.layer0_count.push_back(0);
        graph_.layer0.insert(graph_.layer0.end(), layer0_stride(), 0);
        graph_.upper.emplace_back();
        if (level > 0) {
            graph_.upper.back().resize(level);
        }
        resize_visited(graph_.size());
    }

    // ── Neighbor access ─────────────────────────────────────

    std::span<const Offset> get_neighbors(Offset node, int layer) const {
        if (layer == 0) {
            return {&graph_.layer0[static_cast<std::size_t>(node) * layer0_stride()],
                    static_cast<std::size_t>(graph_.layer0_count[node])};
        }
        return graph_.upper[node][layer - 1];
    }

    void set_neighbors(Offset node, int layer, const std::vector<Offset>& nbs) {
        if (layer == 0) {
            int count = std::min(static_cast<int>(nbs.size()), max_m0());
            Offset* dst = &graph_.layer0[static_cast<std::size_t>(node) * layer0_stride()];
            std::copy_n(nbs.begin(), count, dst);
            graph_.layer0_count[node] = count;
        } else {
            graph_.upper[node][layer - 1] = nbs;
        }
    }

    void add_neighbor(Offset node, int layer, Offset nb) {
        if (layer == 0) {
            int& count = graph_.layer0_count[node];
            graph_.layer0[static_cast<std::size_t>(node) * layer0_stride() + count] = nb;
            count++;
        } else {
            graph_.upper[node][layer - 1].push_back(nb);
        }
    }

    int neighbor_count(Offset node, int layer) const {
        if (layer == 0) {
            return graph_.layer0_count[node];
        }
        return static_cast<int>(graph_.upper[node][layer - 1].size());
    }

    bool has_neighbor(Offset node, int layer, Offset nb) const {
        auto nbs = get_neighbors(node, layer);
        return std::find(nbs.begin(), nbs.end(), nb) != nbs.end();
    }

    void remove_neighbor(Offset node, int layer, Offset nb) {
        if (layer == 0) {
            int& count = graph_.layer0_count[node];
            Offset* base = &graph_.layer0[static_cast<std::size_t>(node) * layer0_stride()];
            for (int i = 0; i < count; i++) {
                if (base[i] == nb) {
                    base[i] = base[count - 1];
                    count--;
                    return;
                }
            }
        } else {
            auto& nbs = graph_.upper[node][layer - 1];
            auto it = std::find(nbs.begin(), nbs.end(), nb);
            if (it != nbs.end()) {
                nbs.erase(it);
            }
        }
    }

    // ── Greedy search ───────────────────────────────────────

    std::vector<Offset> greedy_search(const float* query, int layer, Offset start, int ef) const {
        using Pair = std::pair<float, Offset>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> candidates;
        std::priority_queue<Pair> results;

        resize_visited(graph_.size());
        if (++current_gen_ == 0) {
            std::fill(visited_gen_.begin(), visited_gen_.end(), 0);
            current_gen_ = 1;
        }

        Dim dim = store_.dimensions();

        float d_start = l2_distance(query, store_.get_vector(start), dim);
        candidates.push({d_start, start});
        results.push({d_start, start});
        mark_visited(start);

        while (!candidates.empty()) {
            auto [cd, cid] = candidates.top();
            candidates.pop();

            if (cd > results.top().first) break;

            auto neighbors = get_neighbors(cid, layer);

            for (Offset nb : neighbors) {
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(store_.get_vector(nb), 0, 0);
#endif
            }

            for (Offset nb : neighbors) {
                if (is_visited(nb)) continue;
                mark_visited(nb);

                float d = l2_distance(query, store_.get_vector(nb), dim);

                if (static_cast<int>(results.size()) < ef || d < results.top().first) {
                    candidates.push({d, nb});
                    results.push({d, nb});
                    if (static_cast<int>(results.size()) > ef) {
                        results.pop();
                    }
                }
            }
        }

        std::vector<Pair> sorted;
        sorted.reserve(results.size());
        while (!results.empty()) {
            sorted.push_back(results.top());
            results.pop();
        }
        std::sort(sorted.begin(), sorted.end());

        std::vector<Offset> result;
        result.reserve(sorted.size());
        for (auto& [d, id] : sorted) {
            result.push_back(id);
        }
        return result;
    }

    // ── Neighbor selection heuristic ────────────────────────

    std::vector<Offset> select_neighbors_heuristic(const float* base,
                                                   const std::vector<Offset>& candidates,
                                                   int m) const {
        std::vector<Offset> selected;
        selected.reserve(m);
        Dim dim = store_.dimensions();

        // Diversity-aware: accept only if closer to base than to any selected neighbor.
        for (Offset c : candidates) {
            if (static_cast<int>(selected.size()) >= m) break;

            float dist_to_base = l2_distance(base, store_.get_vector(c), dim);

            bool good = true;
            for (Offset s : selected) {
                if (l2_distance(store_.get_vector(c), store_.get_vector(s), dim) < dist_to_base) {
                    good = false;
                    break;
                }
            }

            if (good) selected.push_back(c);
        }

        // Fill remaining slots.
        for (Offset c : candidates) {
            if (static_cast<int>(selected.size()) >= m) break;
            if (std::find(selected.begin(), selected.end(), c) == selected.end()) {
                selected.push_back(c);
            }
        }

        return selected;
    }

    // ── Visited tracking ────────────────────────────────────

    void resize_visited(std::size_t n) const {
        if (visited_gen_.size() < n) {
            visited_gen_.resize(n, 0);
        }
    }

    bool is_visited(Offset id) const { return visited_gen_[id] == current_gen_; }
    void mark_visited(Offset id) const { visited_gen_[id] = current_gen_; }
};

}  // namespace vexdb