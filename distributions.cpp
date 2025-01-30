#include "distributions.h"

double WGaussian::ApplyToScalar(double x) const {
    if (std::abs(x) > 6 * sigma_competition_radius_) {
        return 0;
    }
    return c_competition_death_ *
           exp(-(x * x / (2 * sigma_competition_radius_ * sigma_competition_radius_))) /
           (sqrt(2 * M_PI) * sigma_competition_radius_);
}

void WGaussian::ApplyToVector(std::vector<double> &source) const {
    for (double &i: source) {
        i = ApplyToScalar(i);
    }
}
double UGaussian::ApplyToScalar(double x) const {
    if (std::abs(x) > 6 * s_dispersal_radius_) {
        return 0;
    }
    return exp(-(x * x / (2 * s_dispersal_radius_ * s_dispersal_radius_))) /
           (sqrt(2 * M_PI) * s_dispersal_radius_);
}

void UGaussian::ApplyToVector(std::vector<double> &source) const {
    for (double &i: source) {
        i = ApplyToScalar(i);
    }
}
