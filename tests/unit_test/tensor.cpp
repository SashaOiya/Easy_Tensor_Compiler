#include "tensor.hpp"

#include <gtest/gtest.h>

TEST(tensor, ctor) {
    Tensor<double> tensor = {2, 2, 2, 2};

    EXPECT_EQ(tensor.batch(), 2);
    EXPECT_EQ(tensor.channels(), 2);
    EXPECT_EQ((tensor[0, 0].size()), 4);
    EXPECT_EQ((tensor[0, 1].size()), 4);
    EXPECT_EQ((tensor[1, 1].size()), 4);
}

TEST(tensor, ctor_init) {
    Tensor<double> tensor = {2, 2, 2, 2};
    tensor[1, 1, 1, 0] = 4;

    EXPECT_EQ((tensor[1, 1, 1, 0]), 4);
}

TEST(tensor, ctor_init_list) {
    Tensor<double> tensor = { 1, 2, {Matrix<double>(2,3), Matrix<double>(2,3)} };
    tensor[0, 1, 1, 0] = 4;

    EXPECT_EQ((tensor[0, 1, 1, 0]), 4);
}

TEST(tensor, ctor_init_list_matrix) {
    Tensor<double> tensor = { 1, 2, {Matrix<double>(1, 2, {5, 5}), Matrix<double>(1, 2, {6, 7})} };
    EXPECT_EQ((tensor[0, 1, 0, 1]), 7);
}

TEST(tensor, copy_ctor) {
    Tensor<double> tensor = { 1, 2, {Matrix<double>(2,1), Matrix<double>(2,1)} };
    Tensor<double> lhs = tensor;

    EXPECT_TRUE(tensor.num_elements() == lhs.num_elements());
    const auto batch_ = lhs.batch(), channels_ = lhs.channels(), height_ = lhs.height(), width_ = lhs.width();
    for (size_t n = 0; n < batch_; ++n)
        for (auto c = 0; c < channels_; ++c)
            for (auto h = 0; h < height_; ++h)
                for (auto w = 0; w < width_; ++w)
                    EXPECT_EQ((tensor[n, c, h, w]), (lhs[n, c, h, w]));
}

TEST(tensor, copy_assignment) {
    Tensor<double> tensor = { 1, 2, {Matrix<double>(2,1), Matrix<double>(2,1)} };
    Tensor<double> lhs = {0, 0};
    lhs = tensor;

    EXPECT_TRUE(tensor.num_elements() == lhs.num_elements());
    const auto batch_ = lhs.batch(), channels_ = lhs.channels(), height_ = lhs.height(), width_ = lhs.width();
    for (size_t n = 0; n < batch_; ++n)
        for (auto c = 0; c < channels_; ++c)
            for (auto h = 0; h < height_; ++h)
                for (auto w = 0; w < width_; ++w)
                    EXPECT_EQ((tensor[n, c, h, w]), (lhs[n, c, h, w]));
}

TEST(tensor, move_ctor) {
    Tensor<double> tensor = { 1, 1, {Matrix<double>(2,1)} };

    std::vector<std::vector<double>> data = {};
    const int cols_ = tensor.height(), rows_ = tensor.width();
    for (size_t i = 0; i < cols_; ++i) {
        std::vector<double> row = {};
        for (auto j = 0; j < rows_; ++j) row.push_back(tensor[0, 0, i, j]);
        data.push_back(row);
    }
    Tensor<double> lhs = std::move(tensor);

    for (size_t i = 0; i < cols_; ++i)
        for (auto j = 0; j < rows_; ++j) EXPECT_EQ(data[i][j], (lhs[0, 0, i, j]));
}

TEST(tensor, move_assignment) {
    Tensor<double> tensor = { 1, 1, {Matrix<double>(2,1)} };

    std::vector<std::vector<double>> data = {};
    const int cols_ = tensor.height(), rows_ = tensor.width();
    for (size_t i = 0; i < cols_; ++i) {
        std::vector<double> row = {};
        for (auto j = 0; j < rows_; ++j) row.push_back(tensor[0, 0, i, j]);
        data.push_back(row);
    }
    Tensor<double> lhs = {};
    lhs = std::move(tensor);

    for (size_t i = 0; i < cols_; ++i)
        for (auto j = 0; j < rows_; ++j) EXPECT_EQ(data[i][j], (lhs[0, 0, i, j]));
}