#pragma once

namespace vextor {

struct HnswBuildParams {
    int m = 16;
    int ef_construction = 200;
};

struct HnswSearchParams {
    int ef_search = 128;
};

}  // namespace vextor
