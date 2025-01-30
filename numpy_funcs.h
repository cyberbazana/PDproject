#pragma once

#include <vector>

std::vector<double> linspace(int start, int end, int num);

size_t GetAbsoluteIndexFromThirdMoment(int64_t i, int64_t j, size_t N);

int64_t GetRelativeIndexForSecondMoment(size_t i, size_t N);

std::pair<int64_t, int64_t> GetPairRelativeIndexForThirdMoment(size_t i, size_t N);