#pragma once

#include <vector>
#include <cmath>

class WGaussian {
public:
    WGaussian(double c_competition_death, double sigma_competition_radius) {
        c_competition_death_ = c_competition_death;
        sigma_competition_radius_ = sigma_competition_radius;
    }

    [[nodiscard]] double ApplyToScalar(double x) const;

    void ApplyToVector(std::vector<double> &source) const;

private:
    double c_competition_death_;
    double sigma_competition_radius_;
};

class UGaussian {
public:
    explicit UGaussian(double s_dispersal_radius) {
        s_dispersal_radius_ = s_dispersal_radius;
    }

    [[nodiscard]] double ApplyToScalar(double x) const;

    void ApplyToVector(std::vector<double> &source) const;

private:
    double s_dispersal_radius_;
};
