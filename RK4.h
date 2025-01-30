#pragma once

#include <memory>
#include "spacial_moments.h"
#include "Eigen/Dense"

class RK4 {
public:
    RK4(double delta_t, SpacialMoments &model, int N, int L) {
        delta_t_ = delta_t;
        model_ = std::make_unique<SpacialMoments>(model);
        N_ = N;
        L_ = L;
    }

    Eigen::VectorXd
    GetValues(const Eigen::VectorXd &prev_y);


private:
    void ApplyToKth(const Eigen::VectorXd &prev_y,
                    Eigen::VectorXd &k, size_t ind);

    void Normalize(Eigen::VectorXd &new_y) const;

    double delta_t_;
    std::unique_ptr<SpacialMoments> model_;
    int N_;
    int L_;
};
