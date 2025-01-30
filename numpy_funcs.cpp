#include <cstdlib>
#include "numpy_funcs.h"

std::vector<double> linspace(int start, int end, int num) {
    std::vector<double> result;
    result.reserve(num);
    long double step = static_cast<double>(end - start) / static_cast<double>(num - 1);
    for (int i = 0; i < num; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}

size_t get_index(int64_t i, int64_t j, size_t N) {
    return (i + N) * (2 * N + 1) + (j + N) + 1 + 2 * N + 1;
}

int64_t get_N_index(size_t i, size_t N) {
    return static_cast<int64_t>(i) - 1 - N;
}

std::pair<int64_t, int64_t> get_N_pair_ind(size_t i, size_t N) {
    return {((i - 1 - 2 * N - 1) / (2 * N + 1)) - N, ((i - 1 - 2 * N - 1) % (2 * N + 1)) - N};
}