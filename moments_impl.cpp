#include <cmath>
#include <vector>
#include "moments_impl.h"

/*
 * Note that ind_i, ind_j, ind_k are always absolute indexes, for example in an array.
 *           i, j, k are always relative ones, from -N to N.
 */

const double eps = std::numeric_limits<double>::epsilon();

/*
 * Counts second moment.
 * Parameters:
 *   i      - index of F_2(x, t).
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 *   N      - number of grid points.
 * Returns:
 *   pair, formated like: double * (y[0] ^ power). If it is not easy to get the power, then power is 0 and double = full number.
 */

std::pair<double, int> get_correct_second_moment(int64_t i, const Eigen::VectorXd &y, size_t N) {
    if (std::abs(i) > N) {
        return {1, 2};
    }
    return {y[1 + i + N], 0};
}

/*
 * Counts third moment.
 * Parameters:
 *   i      - index of F_3(x, y, t), x in our case.
 *   j      - index of F_3(x, y, t)  y in our case.
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 *   N      - number of grid points.
 * Returns:
 *   pair, formated like: double * (y[0] ^ power). If it is not easy to get the power, then power is 0 and double = full number.
 */

std::pair<double, int> get_correct_third_moment(int64_t i, int64_t j, const Eigen::VectorXd &y, size_t N) {
    if (std::abs(i) > N) {
        if (std::abs(j) <= N && std::abs(i - j) <= N) {
            return {y[GetAbsoluteIndexFromThirdMoment(-j, i - j, N)], 0};
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
                return {y[GetAbsoluteIndexFromThirdMoment(j - i, -i, N)], 0};
            } else {
                auto second_i = get_correct_second_moment(i, y, N);
                auto second_j = get_correct_second_moment(j, y, N);
                auto second_abs_diff = get_correct_second_moment(std::abs(i - j), y, N);
                int power = -3 + second_j.second + second_i.second + second_abs_diff.second;
                long double top = (second_i.first * second_j.first * second_abs_diff.first);
                return {top, power};
            }
        } else {
            return {y[GetAbsoluteIndexFromThirdMoment(i, j, N)], 0};
        }
    }
}

/*
 * Counts fourth moment.
 * Parameters:
 *   i      - index of F_4(x, y, z, t), x in our case.
 *   j      - index of F_4(x, y, z, t)  y in our case.
 *   k      - index of F_4(x, y, z, t)  z in our case.
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 *   N      - number of grid points.
 * Returns:
 *   pair, formated like: double * (y[0] ^ power). If it is not easy to get the power, then power is 0 and double = full number.
 */

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
                                                                                               second_j.first *
                                                                                               second_k.first *
                                                                                               second_j_i.first *
                                                                                               second_k_j.first *
                                                                                               second_k_i.first) + eps);
    return {res, pow};

}
