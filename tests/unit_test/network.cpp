#include "network.hpp"

#include <gtest/gtest.h>

#include "operations.hpp"

using KeyT = double;
TEST(network, ScalarAddOperation) {
    Tensor<KeyT> input(1, 2, {Matrix<KeyT>(2, 2, {1, 2, 3, 4}), Matrix<KeyT>(2, 2)});
    Tensor<KeyT> weight(1, 2, {Matrix<KeyT>(2, 2, {10, 20, 30, 40}), Matrix<KeyT>(2, 2)});

    auto input_node = std::make_shared<InputData<KeyT>>(input);
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<ScalarAddOperation<KeyT>>(input_node, weight));

    Tensor output = nn.infer();

    EXPECT_EQ((output[0, 0, 0, 0]), 11);
    EXPECT_EQ((output[0, 0, 0, 1]), 22);
    EXPECT_EQ((output[0, 0, 1, 0]), 33);
    EXPECT_EQ((output[0, 0, 1, 1]), 44);
}

TEST(network, ScalarSubOperation) {
    Tensor<KeyT> input(1, 2, {Matrix<KeyT>(2, 2, {11, 22, 33, 44}), Matrix<KeyT>(2, 2)});
    Tensor<KeyT> weight(1, 2, {Matrix<KeyT>(2, 2, {1, 2, 3, 4}), Matrix<KeyT>(2, 2)});

    auto input_node = std::make_shared<InputData<KeyT>>(input);
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<ScalarSubOperation<KeyT>>(input_node, weight));

    Tensor output = nn.infer();

    EXPECT_EQ((output[0, 0, 0, 0]), 10);
    EXPECT_EQ((output[0, 0, 0, 1]), 20);
    EXPECT_EQ((output[0, 0, 1, 0]), 30);
    EXPECT_EQ((output[0, 0, 1, 1]), 40);
}

TEST(network, ScalarMulOperation) {
    Tensor<KeyT> input(1, 2, {Matrix<KeyT>(2, 2, {10, 20, 30, 40}), Matrix<KeyT>(2, 2)});
    Tensor<KeyT> weight(1, 2, {Matrix<KeyT>(2, 2, {1, 2, 3, 4}), Matrix<KeyT>(2, 2)});

    auto input_node = std::make_shared<InputData<KeyT>>(input);
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<ScalarMulOperation<KeyT>>(input_node, weight));

    Tensor output = nn.infer();

    EXPECT_EQ((output[0, 0, 0, 0]), 10);
    EXPECT_EQ((output[0, 0, 0, 1]), 40);
    EXPECT_EQ((output[0, 0, 1, 0]), 90);
    EXPECT_EQ((output[0, 0, 1, 1]), 160);
}

TEST(network, MatMulOperation) {
    Tensor<KeyT> input(1, 1, {Matrix<KeyT>(2, 2, {1, 2, 3, 4})});
    Tensor<KeyT> weight(1, 1, {Matrix<KeyT>(2, 2, {5, 6, 7, 8})});

    auto input_node = std::make_shared<InputData<KeyT>>(input);
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<MatMulOperation<KeyT>>(input_node, weight));

    Tensor output = nn.infer();

    EXPECT_EQ((output[0, 0, 0, 0]), 19);
    EXPECT_EQ((output[0, 0, 0, 1]), 22);
    EXPECT_EQ((output[0, 0, 1, 0]), 43);
    EXPECT_EQ((output[0, 0, 1, 1]), 50);
}

TEST(network, ConvolOperation) {
    Tensor<KeyT> input(1, 1, {Matrix<KeyT>(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9})});
    Tensor<KeyT> weight(1, 1, {Matrix<KeyT>(2, 2, {1, 0, 0, -1})});

    auto input_node = std::make_shared<InputData<KeyT>>(input);
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<ConvolOperation<KeyT>>(input_node, weight));

    Tensor output = nn.infer();

    EXPECT_EQ((output[0, 0, 0, 0]), -4);
    EXPECT_EQ((output[0, 0, 0, 1]), -4);
    EXPECT_EQ((output[0, 0, 1, 0]), -4);
    EXPECT_EQ((output[0, 0, 1, 1]), -4);
}

TEST(network, ReLuOperation) {
    Tensor<KeyT> input(1, 1, {Matrix<KeyT>(2, 2, {-2.25, 2, -1, 3})});

    auto input_node = std::make_shared<InputData<KeyT>>(input);
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<ReLUOperation<KeyT>>(input_node));

    Tensor output = nn.infer();

    EXPECT_EQ((output[0, 0, 0, 0]), 0);
    EXPECT_EQ((output[0, 0, 0, 1]), 2);
    EXPECT_EQ((output[0, 0, 1, 0]), 0);
    EXPECT_EQ((output[0, 0, 1, 1]), 3);
}

TEST(network, SoftmaxOperation) {
    Tensor<KeyT> input(1, 1, {Matrix<KeyT>(2, 3, {1, 2, 3, 4, 5, 6})});

    auto input_node = std::make_shared<InputData<KeyT>>(input);
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<SoftmaxOperation<KeyT>>(input_node));

    Tensor output = nn.infer();
    const auto eps = 1e-4;

    EXPECT_TRUE(fabs((output[0, 0, 0, 0]) - 0.09) < eps);
    EXPECT_TRUE(fabs((output[0, 0, 0, 1]) - 0.2447) < eps);
    EXPECT_TRUE(fabs((output[0, 0, 0, 2]) - 0.6652) < eps);
    EXPECT_TRUE(fabs((output[0, 0, 1, 0]) - 0.09) < eps);
}