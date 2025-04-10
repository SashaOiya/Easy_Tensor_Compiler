#include "network.hpp"

#include <gtest/gtest.h>

using KeyT = double;
TEST(matrix_buffer, ctor) {
    Tensor<KeyT> input(1, 2, {Matrix<KeyT>(2, 3), Matrix<KeyT>(2, 3)});
    input[0, 0, 0, 0] = 1;
    input[0, 0, 0, 1] = 2;
    input[0, 0, 1, 0] = 3;
    input[0, 0, 1, 1] = 4;

    Tensor<KeyT> weight(1, 2, {Matrix<KeyT>(2, 3), Matrix<KeyT>(2, 3)});
    weight[0, 0, 0, 0] = 10;
    weight[0, 0, 0, 1] = 20;
    weight[0, 0, 1, 0] = 30;
    weight[0, 0, 1, 1] = 40;

    auto input_node = std::make_shared<InputData<KeyT>>(input);

    // Создаём сеть и добавляем операцию сложения
    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<ScalarAddOperation<KeyT>>(input_node, weight));

    Tensor output = nn.infer();
    output.dump();  // Печатаем результат
}