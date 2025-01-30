#pragma once

#include <cstddef>
#include <vector>
#include "distributions.h"
#include "numpy_funcs.h"
#include "Eigen/Dense"

class SpacialMoments {
public:
    SpacialMoments(int N, int L, double h, double q_death, double q_birth) {
        N_ = N;

        std::vector<double> dz_mask(2 * N + 1, 1);

        dz_mask[2 * N] = 1.0 / 3.0;
        for (int64_t i = 2 * N - 1; i >= 1; i -= 2) {
            dz_mask[i] = 4.0 / 3.0;
            dz_mask[i - 1] = 2.0 / 3.0;
        }
        dz_mask[0] = 1.0 / 3.0;

        dz_mask_ = Eigen::Map<const Eigen::VectorXd>(dz_mask.data(), dz_mask.size());
        grid_ = Eigen::Map<const Eigen::VectorXd>(linspace(-L, L, 2 * N + 1).data(), (2 * N + 1));
        h_ = h;
        q_death_ = q_death;
        q_birth_ = q_birth;
        grid_u_ = Eigen::VectorXd::Zero(2 * N + 1);
        grid_w_ = Eigen::VectorXd::Zero(2 * N + 1);
    }

    double
    GetCorrectMoment(const Eigen::VectorXd &y, size_t ind_i);

    void MakeMasks(UGaussian &uGaussian, WGaussian &wGaussian);

    [[nodiscard]] int GetN() const {
        return N_;
    }

private:
    int N_;
    Eigen::VectorXd dz_mask_;
    Eigen::VectorXd grid_;
    double h_;
    double q_death_;
    double q_birth_;
    Eigen::VectorXd grid_u_;
    Eigen::VectorXd grid_w_;


    double GetFirstMoment(const Eigen::VectorXd &y);

    double
    GetSecondMoment(const Eigen::VectorXd &y, size_t ind_i);

    double
    GetThirdMoment(const Eigen::VectorXd &y, size_t ind_i);
};
