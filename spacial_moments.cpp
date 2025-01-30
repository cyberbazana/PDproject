#include "spacial_moments.h"
#include "numpy_funcs.h"
#include "moments_impl.h"

/*
 * Note that ind_i, ind_j, ind_k are always absolute indexes, for example in an array.
 *           i, j, k are always relative ones, from -N to N.
 */

/*
 * Converts from "exponential" representation.
 * Parameters:
 *   pair   - .first is a float part, .second is a power of y[0].
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 * Returns:
 *   Normal representation of "exponential" representation.
 */

double Converter(std::pair<double, int> &&pair, const Eigen::VectorXd &y) {
    return (pair.first * std::pow(y(0), pair.second));
}

/*
 * Counts first moment as in the article.
 * Parameters:
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 * Returns:
 *   First moment as in the article.
 */

double SpacialMoments::GetFirstMoment(const Eigen::VectorXd &y) {
    double ans = (q_birth_ - q_death_) * y[0];
    double sum = 0.0;
    double c = 0.0;
    for (size_t ind_i = 1; ind_i < 2 * N_ + 2; ++ind_i) {
        double y_d = grid_w_[ind_i - 1] * y[ind_i] * dz_mask_[ind_i - 1] - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    ans -= sum;
    return ans;
}

/*
 * Counts second moment as in the article.
 * Parameters:
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 *   ind_i  - from 1 to 2 * N + 1, where 1 = -Nth second moment, ... 2 * N + 1 = Nth second moment.
 * Returns:
 *   Second moment as in the article.
 */

double SpacialMoments::GetSecondMoment(const Eigen::VectorXd &y, size_t ind_i) {
    int64_t i = GetRelativeIndexForSecondMoment(ind_i, N_);
    double ans = -2 * (q_death_ + grid_w_[ind_i - 1]) * y[ind_i] +
                 2 * grid_u_[ind_i - 1] * q_birth_ * y[0];
    double sum = 0.0;
    double c = 0.0;
    for (size_t ind_j = 0; ind_j < 2 * N_ + 1; ++ind_j) {
        int64_t j = GetRelativeIndexForSecondMoment(ind_j, N_) + 1;
        double y_d = (grid_u_[ind_j] * q_birth_ *
                      (Converter(get_correct_second_moment(i + j, y, N_), y) +
                       Converter(get_correct_second_moment(i - j, y, N_), y)) -
                      grid_w_[ind_j] *
                      (Converter(get_correct_third_moment(i, j, y, N_), y) +
                       Converter(get_correct_third_moment(-i, j, y, N_), y))) * dz_mask_[ind_j] - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    ans += sum;
    return ans;
}

/*
 * Counts third moment as in the article.
 * Parameters:
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 *   ind_i  - from 2 * N + 2 to end, where 2 * N + 2 = -Nth -Nth third moment, ... 4 * N + 2 = -Nth Nth third moment, ...
 * Returns:
 *   Third moment as in the article.
 */

double SpacialMoments::GetThirdMoment(const Eigen::VectorXd &y, size_t ind_i) {
    int64_t i = GetPairRelativeIndexForThirdMoment(ind_i, N_).first;
    int64_t j = GetPairRelativeIndexForThirdMoment(ind_i, N_).second;
    double ans = -(3 * q_death_ + 2 * grid_w_[i + N_] +
                   2 * grid_w_[j + N_]);
    if (std::abs(i - j) <= N_) {
        ans -= 2 * grid_w_[i - j + N_];
    }
    ans *= Converter(get_correct_third_moment(i, j, y, N_), y);


    ans += q_birth_ * (grid_u_[i + N_] * (Converter(get_correct_second_moment(j - i, y, N_), y) +
                                          Converter(get_correct_second_moment(j, y, N_), y)) +
                       grid_u_[j + N_] * (Converter(get_correct_second_moment(i, y, N_), y) +
                                          Converter(get_correct_second_moment(j - i, y, N_),
                                                    y)));
    if (std::abs(i - j) <= N_) {
        ans += q_birth_ *
               (grid_u_[i - j + N_] * (Converter(get_correct_second_moment(i, y, N_), y) +
                                       Converter(get_correct_second_moment(j, y, N_), y)));
    }
    double sum = 0.0;
    double c = 0.0;
    for (size_t ind_k = 0; ind_k < 2 * N_ + 1; ++ind_k) {
        int64_t k = GetRelativeIndexForSecondMoment(ind_k, N_) + 1;
        double y_d = (grid_w_[ind_k]) *
                     (Converter(get_correct_fourth_moment(i, j, k, y, N_), y) +
                      Converter(get_correct_fourth_moment(-i, j - i, k, y,
                                                          N_), y) +
                      Converter(get_correct_fourth_moment(-j, i - j, k, y,
                                                          N_), y)) *
                     dz_mask_[ind_k] - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    ans -= sum;
    sum = 0;
    c = 0;
    for (size_t ind_k = 0; ind_k < 2 * N_ + 1; ++ind_k) {
        int64_t k = GetRelativeIndexForSecondMoment(ind_k, N_) + 1;

        double y_d = (grid_u_[ind_k] *
                      (Converter(get_correct_third_moment(k + i, k + j, y, N_), y) +
                       Converter(get_correct_third_moment(k - i, k + j - i, y, N_), y) +
                       Converter(get_correct_third_moment(k - j, k + i - j, y, N_), y))) *
                     dz_mask_[ind_k] - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    sum *= q_birth_;
    ans += sum;
    return ans;
}

/*
 * Gets correct moment for given index.
 * Parameters:
 *   y      - previous iteration values, y[0] = F_1(t), y[1 -- 2 * N + 1] = F_2(x, t), y[2 * N + 2, ...] = F_3(x, y, t) in such order:
 *              (-N -N), (-N -N + 1), ... (-N, N), (-N + 1, -N), ... (N, N).
 *   ind_i  - form 0 to end, there 0 = F_1(t), [1 -- 2 * N + 1] = F_2(x, t), [2 * N + 2, ...] = F_3(x, y, t).
 * Returns:
 *   Correct moment as in the article.
 */

double SpacialMoments::GetCorrectMoment(const Eigen::VectorXd &y, size_t ind_i) {
    if (ind_i == 0) {
        return GetFirstMoment(y);
    }
    if (ind_i <= 2 * N_ + 1) {
        return GetSecondMoment(y, ind_i);
    }
    return GetThirdMoment(y, ind_i);
}

/*
 * Function to prerender values for grid points.
 * Parameters:
 *   uGaussian  - uGaussian distribution.
 *   wGaussian  - wGaussian distribution.
 * Makes:
 *   grid_u_ = grid points with uGaussian distribution.
 *   grid_w_ = grid points with wGaussian distribution.
 */

void SpacialMoments::MakeMasks(UGaussian &uGaussian, WGaussian &wGaussian) {
    for (size_t ind_i = 0; ind_i < 2 * N_ + 1; ++ind_i) {
        grid_u_[ind_i] = uGaussian.ApplyToScalar(grid_[ind_i]);
        grid_w_[ind_i] = wGaussian.ApplyToScalar(grid_[ind_i]);
    }
}
