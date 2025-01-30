#include <cmath>
#include <vector>
#include "moments_impl.h"


const double eps = std::numeric_limits<double>::epsilon();

std::pair<double, int> get_correct_second_moment(int64_t i, const Eigen::VectorXd &y, size_t N) {
    if (std::abs(i) > N) {
        return {1, 2};
    }
    return {y(1 + i + N), 0};
}

std::pair<double, int> get_correct_third_moment(int64_t i, int64_t j, const Eigen::VectorXd &y, size_t N) {
    if (std::abs(i) > N) {
        if (std::abs(j) <= N && std::abs(i - j) <= N) {
            return {y(get_index(-j, i - j, N)), 0};
        } else {
            auto second_i = get_correct_second_moment(i, y, N);
            auto second_j = get_correct_second_moment(j, y, N);
            auto second_abs_diff = get_correct_second_moment(std::abs(i - j), y, N);
            int power = -3 + second_j.second + second_i.second + second_abs_diff.second;
            long double top = (second_i.first * second_j.first * second_abs_diff.first);
            return {top, power};
        }
    } else {
        if (std::abs(j) > N) {
            if (std::abs(i - j) <= N) {
                return {y(get_index(j - i, -i, N)), 0};
            } else {
                auto second_i = get_correct_second_moment(i, y, N);
                auto second_j = get_correct_second_moment(j, y, N);
                auto second_abs_diff = get_correct_second_moment(std::abs(i - j), y, N);
                int power = -3 + second_j.second + second_i.second + second_abs_diff.second;
                long double top = (second_i.first * second_j.first * second_abs_diff.first);
                return {top, power};
            }
        } else {
            return {y(get_index(i, j, N)), 0};
        }
    }
}

std::pair<double, int>
get_correct_fourth_moment(int64_t i, int64_t j, int64_t k, const Eigen::VectorXd &y, size_t N) {
    auto third_ij = get_correct_third_moment(i, j, y, N);
    auto third_ik = get_correct_third_moment(i, k, y, N);
    auto third_jk = get_correct_third_moment(j, k, y, N);
    auto third_j_i = get_correct_third_moment(j - i, k - i, y, N);

    auto second_i = get_correct_second_moment(i, y, N);
    auto second_j = get_correct_second_moment(j, y, N);
    auto second_k = get_correct_second_moment(k, y, N);
    auto second_j_i = get_correct_second_moment(j - i, y, N);
    auto second_k_i = get_correct_second_moment(k - i, y, N);
    auto second_k_j = get_correct_second_moment(k - j, y, N);
    int pow = 4 + third_ij.second + third_ik.second + third_jk.second + third_j_i.second - second_i.second -
              second_j.second - second_k.second - second_j_i.second - second_k_j.second - second_k_i.second;
    long double res = (third_ij.first * third_ik.first * third_jk.first * third_j_i.first) / ((second_i.first *
                 second_j.first * second_k.first * second_j_i.first * second_k_j.first * second_k_i.first) + eps);
    return {res, pow};

}
