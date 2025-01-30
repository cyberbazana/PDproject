#pragma once

#include <vector>
#include <cstdint>
#include "numpy_funcs.h"
#include "Eigen/Dense"


std::pair<double, int> get_correct_second_moment(int64_t i, const Eigen::VectorXd &y, size_t N);

std::pair<double, int> get_correct_third_moment(int64_t i, int64_t j, const Eigen::VectorXd &y, size_t N);

std::pair<double, int> get_correct_fourth_moment(int64_t i, int64_t j, int64_t k, const Eigen::VectorXd &y, size_t N);
