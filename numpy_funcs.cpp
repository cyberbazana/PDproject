#include <cstdlib>
#include "numpy_funcs.h"

/*
 * Comfortable linspace.
 * Parameters:
 *   start  - start.
 *   end    - end.
 *   num    - number of grid points, including start and end.
 * Returns:
 *   Linspace from [start, end], num points in the vector.
 */

std::vector<double> linspace(int start, int end, int num) {
    std::vector<double> result;
    result.reserve(num);
    long double step = static_cast<double>(end - start) / static_cast<double>(num - 1);
    for (int i = 0; i < num; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}

/*
 * Transforms relative index to absolute.
 * Parameters:
 *   i      - x in F_3(x, y, t).
 *   j      - y in F_3(x, y, t).
 *   N      - number of grid points.
 * Returns:
 *   Absolute index to find F_3(x, y, t) in an array.
 */

size_t GetAbsoluteIndexFromThirdMoment(int64_t i, int64_t j, size_t N) {
    return (i + N) * (2 * N + 1) + (j + N) + 1 + 2 * N + 1;
}

/*
 * Transforms absolute index to relative.
 * Parameters:
 *   ind_i  - absolute index from [1, 2 * N + 1]
 *   N      - number of grid points.
 * Returns:
 *   Relative index for calculations.
 */

int64_t GetRelativeIndexForSecondMoment(size_t i, size_t N) {
    return static_cast<int64_t>(i) - 1 - N;
}

/*
 * Transforms absolute index a pair of relative ones.
 * Parameters:
 *   ind_i  - absolute index from [2 * N + 2, ...]
 *   N      - number of grid points.
 * Returns:
 *   A pair of relative indexed for calculations. For example, 2 * N + 2 --> (-N, -N)
 */

std::pair<int64_t, int64_t> GetPairRelativeIndexForThirdMoment(size_t i, size_t N) {
    return {((i - 1 - 2 * N - 1) / (2 * N + 1)) - N, ((i - 1 - 2 * N - 1) % (2 * N + 1)) - N};
}