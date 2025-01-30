#include "spacial_moments.h"
#include "numpy_funcs.h"
#include "moments_impl.h"

double Converter(std::pair<double, int> &&pair, const Eigen::VectorXd &y) {
    return (pair.first * std::pow(y(0), pair.second));
}

double SpacialMoments::GetFirstMoment(const Eigen::VectorXd &y) {
    double ans = (q_birth_ - q_death_) * y(0);
    double sum = 0.0;
    double c = 0.0;
    for (size_t i = 1; i < 2 * N_ + 2; ++i) {
        double y_d = grid_w_(i - 1) * y(i) * dz_mask_(i - 1) - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    ans -= sum;
    return ans;
}

double SpacialMoments::GetSecondMoment(const Eigen::VectorXd &y, size_t ind_i) {
    int64_t i = get_N_index(ind_i, N_);
    double ans = -2 * (q_death_ + grid_w_(ind_i - 1)) * y(ind_i) +
                 2 * grid_u_(ind_i - 1) * q_birth_ * y(0);
    double sum = 0.0;
    double c = 0.0;
    for (size_t ind_j = 0; ind_j < 2 * N_ + 1; ++ind_j) {
        int64_t j = get_N_index(ind_j, N_) + 1;
        double y_d =  (grid_u_(ind_j) * q_birth_ *
                           (Converter(get_correct_second_moment(i + j, y, N_), y) +
                            Converter(get_correct_second_moment(i - j, y, N_), y)) -
                           grid_w_(ind_j) *
                           (Converter(get_correct_third_moment(i, j, y, N_), y) +
                            Converter(get_correct_third_moment(-i, j, y, N_), y))) * dz_mask_(ind_j) - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    ans += sum;
    return ans;
}

double SpacialMoments::GetThirdMoment(const Eigen::VectorXd &y, size_t ind_i) {
    int64_t i = get_N_pair_ind(ind_i, N_).first;
    int64_t j = get_N_pair_ind(ind_i, N_).second;
    double ans = -(3 * q_death_ + 2 * grid_w_(i + N_) +
                   2 * grid_w_(j + N_));
    if (std::abs(i - j) <= N_) {
        ans -= 2 * grid_w_(i - j + N_);
    }
    ans *= Converter(get_correct_third_moment(i, j, y, N_), y);


    ans += q_birth_ * (grid_u_(i + N_) * (Converter(get_correct_second_moment(j - i, y, N_), y) +
                                          Converter(get_correct_second_moment(j, y, N_), y)) +
                       grid_u_(j + N_) * (Converter(get_correct_second_moment(i, y, N_), y) +
                                          Converter(get_correct_second_moment(j - i, y, N_),
                                                    y)));
    if (std::abs(i - j) <= N_) {
        ans += q_birth_ *
               (grid_u_(i - j + N_) * (Converter(get_correct_second_moment(i, y, N_), y) +
                                       Converter(get_correct_second_moment(j, y, N_), y)));
    }
    double sum = 0.0;
    double c = 0.0;
    for (size_t ind_k = 0; ind_k < 2 * N_ + 1; ++ind_k) {
        int64_t k = get_N_index(ind_k, N_) + 1;
        double y_d =  (grid_w_(ind_k)) *
                     (Converter(get_correct_fourth_moment(i, j, k, y, N_), y) +
                      Converter(get_correct_fourth_moment(-i, j - i, k, y,
                                                          N_), y) +
                      Converter(get_correct_fourth_moment(-j, i - j, k, y,
                                                          N_), y)) *
                     dz_mask_(ind_k) - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    ans -= sum;
    sum = 0;
    c = 0;
    for (size_t ind_k = 0; ind_k < 2 * N_ + 1; ++ind_k) {
        int64_t k = get_N_index(ind_k, N_) + 1;

        double y_d = (grid_u_(ind_k) *
                           (Converter(get_correct_third_moment(k + i, k + j, y, N_), y) +
                            Converter(get_correct_third_moment(k - i, k + j - i, y, N_), y) +
                            Converter(get_correct_third_moment(k - j, k + i - j, y, N_), y))) *
                     dz_mask_(ind_k) - c;
        double t_d = sum + y_d;
        c = (t_d - sum) - y_d;
        sum = t_d;
    }
    sum *= h_;
    sum *= q_birth_;
    ans += sum;
    return ans;
}

double SpacialMoments::GetCorrectMoment(const Eigen::VectorXd &prev_y, size_t i) {
    if (i == 0) {
        return GetFirstMoment(prev_y);
    }
    if (i <= 2 * N_ + 1) {
        return GetSecondMoment(prev_y, i);
    }
    return GetThirdMoment(prev_y, i);
}

void SpacialMoments::MakeMasks(UGaussian &uGaussian, WGaussian &wGaussian) {
    for (size_t i = 0; i < 2 * N_ + 1; ++i) {
        grid_u_[i] = uGaussian.ApplyToScalar(grid_[i]);
        grid_w_[i] = wGaussian.ApplyToScalar(grid_[i]);
    }
}
