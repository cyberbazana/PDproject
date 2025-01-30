#include <thread>
#include <iostream>
#include "RK4.h"
#include "numpy_funcs.h"

const double eps = std::numeric_limits<double>::epsilon();

Eigen::VectorXd RK4::GetValues(const Eigen::VectorXd &prev_y) {
    size_t size = 1 + (2 * N_ + 1) + (2 * N_ + 1) * (2 * N_ + 1);
    Eigen::VectorXd k_1 = Eigen::VectorXd::Zero(size);
    Eigen::VectorXd k_2 = Eigen::VectorXd::Zero(size);
    Eigen::VectorXd k_3 = Eigen::VectorXd::Zero(size);
    Eigen::VectorXd k_4 = Eigen::VectorXd::Zero(size);
    Eigen::VectorXd ans = Eigen::VectorXd::Zero(size);
    size_t number_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk = size / number_threads;
    auto op = [this](Eigen::VectorXd &k_i, const Eigen::VectorXd &src,
                     size_t start,
                     size_t stop) {
        for (size_t i = start; i < stop; ++i) {
            ApplyToKth(src, k_i, i);
        }
    };

    for (size_t i = 0; i < number_threads; ++i) {
        threads.emplace_back(op, std::ref(k_1), std::ref(prev_y), i * chunk, (i + 1) * chunk);
    }
    for (auto &t: threads) {
        t.join();
    }
    if (chunk * number_threads < size) {
        for (size_t i = chunk * number_threads; i < size; ++i) {
            ApplyToKth(prev_y, k_1, i);
        }
    }
    threads.clear();
    Eigen::VectorXd k_1_vector_divided_by_two = k_1 * 0.5;
    Eigen::VectorXd sum_for_k_2 = k_1_vector_divided_by_two + prev_y;

    for (size_t i = 0; i < number_threads; ++i) {
        threads.emplace_back(op, std::ref(k_2), std::ref(sum_for_k_2), i * chunk, (i + 1) * chunk);
    }
    for (auto &t: threads) {
        t.join();
    }
    if (chunk * number_threads < size) {
        for (size_t i = chunk * number_threads; i < size; ++i) {
            ApplyToKth(sum_for_k_2, k_2, i);
        }
    }
    threads.clear();

    Eigen::VectorXd k_2_vector_divided_by_two = k_2 * 0.5;
    Eigen::VectorXd sum_for_k_3 =
            k_2_vector_divided_by_two + prev_y;

    for (size_t i = 0; i < number_threads; ++i) {
        threads.emplace_back(op, std::ref(k_3), std::ref(sum_for_k_3), i * chunk, (i + 1) * chunk);
    }
    for (auto &t: threads) {
        t.join();
    }
    if (chunk * number_threads < size) {
        for (size_t i = chunk * number_threads; i < size; ++i) {
            ApplyToKth(sum_for_k_3, k_3, i);
        }
    }
    threads.clear();

    Eigen::VectorXd sum_for_k_4 = k_3 + prev_y;
    for (size_t i = 0; i < number_threads; ++i) {
        threads.emplace_back(op, std::ref(k_4), std::ref(sum_for_k_4), i * chunk, (i + 1) * chunk);
    }
    for (auto &t: threads) {
        t.join();
    }
    if (chunk * number_threads < size) {
        for (size_t i = chunk * number_threads; i < size; ++i) {
            ApplyToKth(sum_for_k_4, k_4, i);
        }
    }
    threads.clear();

    ans = prev_y + ((k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6.0);
    Normalize(ans);
    return ans;
}

void RK4::Normalize(Eigen::VectorXd &new_y) const {
    size_t size = 1 + (2 * N_ + 1) + (2 * N_ + 1) * (2 * N_ + 1);
    for (size_t i = 1; i < size; ++i) {
        if (i <= 2 * N_ + 1) {
            if (std::abs(new_y[i] - new_y[2 * N_ + 1 - i + 1]) > eps) {
                double mid = (new_y[i] + new_y[2 * N_ + 1 - i + 1]) / 2;
                new_y[i] = mid;
                new_y[2 * N_ + 1 - i + 1] = mid;
            }
        }
        else {

        }
    }
}

void RK4::ApplyToKth(const Eigen::VectorXd &prev_y, Eigen::VectorXd &k, size_t ind) {
    if (ind <= 2 * N_ + 1) {
        k[ind] = delta_t_ * model_->GetCorrectMoment(prev_y, ind);
    } else {
        int64_t i = get_N_pair_ind(ind, N_).first;
        int64_t j = get_N_pair_ind(ind, N_).second;
        if (k(get_index(-i, -j, N_)) != 0.0) {
            k[ind] = k[get_index(-i, -j, N_)];
            return;
        }
        if (k(get_index(-j, -i, N_)) != 0.0) {
            k[ind] = k[get_index(-j, -i, N_)];
            return;
        }
        if (k(get_index(j, i, N_)) != 0.0) {
            k[ind] = k[get_index(j, i, N_)];
            return;
        }
        if (std::abs(i - j) <= N_ && k(get_index(-i, j - i, N_)) != 0.0) {
            k[ind] = k[get_index(-i, j - i, N_)];
            return;
        }
        if (std::abs(i - j) <= N_ && k(get_index(j - i, -i, N_)) != 0.0) {
            k[ind] = k(get_index(j - i, -i, N_));
            return;
        }
        if (std::abs(i - j) <= N_ && k(get_index(-j, i - j, N_)) != 0.0) {
            k[ind] = k(get_index(-j, i - j, N_));
            return;
        }
        if (std::abs(i - j) <= N_ && k(get_index(i - j, -j, N_)) != 0.0) {
            k[ind] = k(get_index(i - j, -j, N_));
            return;
        }
        k(ind) = delta_t_ * model_->GetCorrectMoment(prev_y, ind);
    }
}
