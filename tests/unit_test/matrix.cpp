#include "matrix.hpp"

#include <gtest/gtest.h>

TEST(matrix_buffer, ctor) {
    Matrix<double> matrix = {2, 3, {1, 2, 3, 4, 6, 7}};
    Matrix<double> lhs = matrix;

    auto it = matrix.begin();
    EXPECT_EQ(*(++it), 2);
}

TEST(matrix, copy_assignment) {
    Matrix<double> matrix = {3, 3, {-4, 9, 6, 4, -11, 0, 7, 4, 5}};
    Matrix<double> lhs = {0, 0};
    lhs = matrix;

    EXPECT_TRUE(matrix.size() == lhs.size());
    auto cols_ = matrix.n_cols(), rows_ = matrix.n_rows();
    for (size_t i = 0; i < cols_; ++i)
        for (auto j = 0; j < rows_; ++j) EXPECT_EQ((matrix[i, j]), (lhs[i, j]));
}

TEST(matrix, move_ctor) {
    Matrix<double> matrix = {3, 3, {-4, 9, 6, 4, -11, 0, 7, 4, 5}};

    std::vector<std::vector<double>> data = {};
    const int cols_ = matrix.n_cols(), rows_ = matrix.n_rows();
    for (size_t i = 0; i < cols_; ++i) {
        std::vector<double> row = {};
        for (auto j = 0; j < rows_; ++j) row.push_back(matrix[i, j]);
        data.push_back(row);
    }
    Matrix<double> lhs = std::move(matrix);

    for (size_t i = 0; i < cols_; ++i)
        for (auto j = 0; j < rows_; ++j) EXPECT_EQ(data[i][j], (lhs[i, j]));
}

TEST(matrix, move_assignment) {
    Matrix<double> matrix = {3, 3, {5, 8, 9, 12, 3, -4, 67, 18, -5}};

    std::vector<std::vector<double>> data = {};
    const int cols_ = matrix.n_cols(), rows_ = matrix.n_rows();
    for (size_t i = 0; i < cols_; ++i) {
        std::vector<double> row = {};
        for (auto j = 0; j < rows_; ++j) row.push_back(matrix[i, j]);
        data.push_back(row);
    }

    Matrix<double> lhs = {0, 0};
    lhs = std::move(matrix);

    for (size_t i = 0; i < cols_; ++i)
        for (auto j = 0; j < rows_; ++j) EXPECT_EQ(data[i][j], (lhs[i, j]));
}

TEST(matrix_buffer, multiply) {
    Matrix<double> matrix = {2, 3, {1, 2, 3, 4, 6, 7}};
    Matrix<double> lhs = {3, 2, {3, 5, 7, 1, 2, 3}};

    Matrix<double> mul = multiply<double>(matrix, lhs);

    EXPECT_EQ((mul[0, 0]), 23);
    EXPECT_EQ((mul[0, 1]), 16);
    EXPECT_EQ((mul[1, 0]), 68);
    EXPECT_EQ((mul[1, 1]), 47);
}