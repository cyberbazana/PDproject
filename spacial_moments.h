#pragma once

#include <cstddef>
#include <vector>
#include "distributions.h"
#include "Eigen/Dense"

class SpacialMoments {
public:
    SpacialMoments(int N, const Eigen::VectorXd &dz_mask, const Eigen::VectorXd &grid,
                   double h, double q_death,
                   double q_birth) {
        N_ = N;
        dz_mask_ = dz_mask;
        grid_ = grid;
        h_ = h;
        q_death_ = q_death;
        q_birth_ = q_birth;
        grid_u_ = Eigen::VectorXd::Zero(2 * N + 1);
        grid_w_ = Eigen::VectorXd::Zero(2 * N + 1);
    }

    double
    GetCorrectMoment(const Eigen::VectorXd &prev_y, size_t i);

    void MakeMasks(UGaussian &uGaussian, WGaussian &wGaussian);


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
