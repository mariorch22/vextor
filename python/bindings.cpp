#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "segment/segment_manager.h"

namespace nb = nanobind;

NB_MODULE(vexdb, m) {
    m.doc() = "vexdb — segmented vector database for ANN search";

    nb::class_<vexdb::SegmentManager>(m, "Database")
        .def(
            "__init__",
            [](vexdb::SegmentManager* self, vexdb::Dim dimensions, std::size_t segment_capacity,
               const std::string& path) {
                new (self) vexdb::SegmentManager(dimensions, segment_capacity, path);
            },
            nb::arg("dimensions"), nb::arg("segment_capacity"), nb::arg("path") = "")

        .def(
            "insert",
            [](vexdb::SegmentManager& self, vexdb::VectorId user_id,
               nb::ndarray<float, nb::ndim<1>, nb::c_contig> vec) {
                self.insert(user_id, {vec.data(), vec.shape(0)});
            },
            nb::arg("user_id"), nb::arg("vector").noconvert())

        .def(
            "insert_batch",
            [](vexdb::SegmentManager& self, nb::ndarray<uint64_t, nb::ndim<1>, nb::c_contig> ids,
               nb::ndarray<float, nb::ndim<2>, nb::c_contig> vecs) {
                if (ids.shape(0) != vecs.shape(0))
                    throw std::invalid_argument("user_ids and vectors must have same length");
                auto n = ids.shape(0);
                auto dim = self.dimensions();
                if (vecs.shape(1) != dim)
                    throw std::invalid_argument("vector dimensions must match database");
                for (std::size_t i = 0; i < n; i++) {
                    self.insert(ids(i), {&vecs(i, 0), dim});
                }
            },
            nb::arg("user_ids").noconvert(), nb::arg("vectors").noconvert())

        .def(
            "search",
            [](const vexdb::SegmentManager& self,
               nb::ndarray<float, nb::ndim<1>, nb::c_contig> query, std::size_t k) {
                auto results = self.search({query.data(), query.shape(0)}, k);
                nb::list out;
                for (const auto& r : results) {
                    out.append(nb::make_tuple(r.user_id, r.distance));
                }
                return out;
            },
            nb::arg("query").noconvert(), nb::arg("k"))

        .def("save", &vexdb::SegmentManager::save)

        .def_static("load", &vexdb::SegmentManager::load, nb::arg("path"))

        .def_prop_ro("size", &vexdb::SegmentManager::total_vectors)
        .def_prop_ro("dimensions", &vexdb::SegmentManager::dimensions)
        .def_prop_ro("segment_count", &vexdb::SegmentManager::segment_count);
}