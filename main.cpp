#include <iostream>
#include <numeric>
#include <fstream>
#include "numpy_funcs.h"
#include "distributions.h"
#include "RK4.h"
#include "spacial_moments.h"
#include "Eigen/Dense"

const int L = 2;
const int N = 500;
const double q_birth = 0.6;
const double q_death = 0.1;
const double h = static_cast<double>(L) / static_cast<double>(N);
const double s_dispersal_radius = 0.02;
const double sigma_competition_radius = 0.02;
const double c_competition_death = 0.01;
const double delta_t = 0.05;
const size_t size = 1 + (2 * N + 1) + (2 * N + 1) * (2 * N + 1);
const double xi2 = 1;
const double xi3 = 1;

int main() {
    std::ofstream outFile("data.csv");

    auto uGaussian = UGaussian(s_dispersal_radius);
    auto wGaussian = WGaussian(c_competition_death, sigma_competition_radius);
    std::vector<double> grid = linspace(-L, L, 2 * N + 1);
    std::vector<double> dz_mask(2 * N + 1, 1);

    dz_mask[2 * N] = 1.0 / 3.0;
    for (int64_t i = 2 * N - 1; i >= 1; i -= 2) {
        dz_mask[i] = 4.0 / 3.0;
        dz_mask[i - 1] = 2.0 / 3.0;
    }
    dz_mask[0] = 1.0 / 3.0;

    std::vector<double> init_y(size, std::pow(6.38, 3) * xi3);
    init_y[0] = 6.38;
    for (size_t i = 1; i < 2 * N + 2; ++i) {
        init_y[i] = std::pow(init_y[0], 2) * xi2;
    }
    Eigen::VectorXd eigen_grid = Eigen::Map<const Eigen::VectorXd>(grid.data(), grid.size());
    Eigen::VectorXd eigen_dz_mask = Eigen::Map<const Eigen::VectorXd>(dz_mask.data(), dz_mask.size());
    Eigen::VectorXd eigen_init = Eigen::Map<const Eigen::VectorXd>(init_y.data(), init_y.size());
    SpacialMoments model(N, eigen_dz_mask, eigen_grid, h, q_death, q_birth);
    model.MakeMasks(uGaussian, wGaussian);
    RK4 rk4(delta_t, model, N, L);
    double st = 0.0;
    for (size_t i = 0; i < 1600; ++i) {
        Eigen::VectorXd res = rk4.GetValues(eigen_init);
        st += delta_t;
        eigen_init = std::move(res);
        std::cout << st << " " << eigen_init[0] << '\n';
    }
    outFile.close();
    return 0;
}
