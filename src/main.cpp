#include <unistd.h>
#include <memory>
#include <stdexcept>

#include "network.hpp"

using KeyT = double;

static std::size_t read_sizes() {
    std::size_t size;
    std::cin >> size;
    if (!std::cin.good()) {
        throw std::runtime_error("Invalid input NCHW");
    }
    return size;
}

static auto read_tensor() {
    std::size_t batches = read_sizes();
    std::size_t channels = read_sizes();
    std::size_t height = read_sizes();
    std::size_t width = read_sizes();

    Tensor<KeyT> input(batches, channels, height, width);

    for (std::size_t batch = 0; batch < batches; ++batch)
        for (std::size_t channel = 0; channel < channels; ++channel)
            for (std::size_t i = 0; i < height; ++i)
                for (std::size_t j = 0; j < width; ++j) {
                    std::cin >> input[batch, channel, i, j];
                    if (!std::cin.good()) {
                        throw std::runtime_error("Invalid input tensor");
                    }
                }
    return input;
}

int main() try{
    Tensor<KeyT> input = read_tensor();

    Tensor<KeyT> weight = read_tensor();

    auto input_node = std::make_shared<InputData<KeyT>>(input);

    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<ScalarAddOperation<KeyT>>(input_node, weight));

    Tensor output = nn.infer();
    output.print();  

    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error : " << e.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Caught unknown exception\n";
    return EXIT_FAILURE;
}