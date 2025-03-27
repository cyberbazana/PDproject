#include <iostream>
#include <numeric>
#include <fstream>
#include "distributions.h"
#include "RK4.h"
#include "spacial_moments.h"
#include "Eigen/Dense"

const int L = 12;
const int N = 600;
const double q_birth = 1;
const double q_death = 0.01;
const double h = static_cast<double>(L) / static_cast<double>(N);
const double s_dispersal_radius = 1;
const double sigma_competition_radius = 0.1;
const double c_competition_death = 1;
const double delta_t = 0.05;
const size_t size = 1 + (2 * N + 1) + (2 * N + 1) * (2 * N + 1);
const double xi2 = 1;
const double xi3 = 1;
const double tau = 300;

const int total_time = 20000;


int main() {
    std::ofstream outFile_first("dataset_2a_first_moments.csv");
    std::ofstream outFile_second("dataset_2a_second_moments.csv");
    std::ofstream outFile_third("dataset_2a_third_moments.csv");
    outFile_first << "first_moment" << '\n';
    outFile_second << "second_moment" << '\n';
    outFile_third << "third_moment" << '\n';

    auto uGaussian = UGaussian(s_dispersal_radius);
    auto wGaussian = WGaussian(c_competition_death, sigma_competition_radius);

    std::vector<double> init_y(size, std::pow(0.4, 3) * xi3);
    init_y[0] = 0.4;
    for (size_t i = 1; i < 2 * N + 2; ++i) {
        init_y[i] = std::pow(init_y[0], 2) * xi2;
    }
    Eigen::VectorXd eigen_init = Eigen::Map<const Eigen::VectorXd>(init_y.data(), init_y.size());
    SpacialMoments model(N, L, q_death, q_birth);
    model.MakeMasks(uGaussian, wGaussian);
    RK4 rk4(delta_t, model);
    double st = 0.0;
    bool fl = true;
    for (size_t i = 0; i < total_time; ++i) {
        Eigen::VectorXd res = rk4.GetValues(eigen_init);
        st += delta_t;
        eigen_init = std::move(res);
        outFile_first << eigen_init[0] << '\n';
        std::cout << st << " " << eigen_init[2 * N + 2] << " " << eigen_init[0] << '\n';
        if (st > tau && fl) {
            for (size_t j = 1; j <= 2 * N + 1; ++j) {
                outFile_second << eigen_init[j] << '\n';
            }
            for (size_t j = 2 * N + 2; j < size; ++j) {
                outFile_third << eigen_init[j] << '\n';
            }
            fl = false;
            outFile_second.close();
            outFile_third.close();
        }
    }
    outFile_first.close();

    return 0;
}
