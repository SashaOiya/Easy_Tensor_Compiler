#include <benchmark/benchmark.h>
#include "network.hpp"
#include <memory>
#include <stdexcept>
#include <iostream>

using KeyT = double;

static Tensor<KeyT> input;
static Tensor<KeyT> weight;
static bool data_loaded = false;

static std::size_t read_sizes() {
    std::size_t size;
    std::cin >> size;
    if (!std::cin.good()) throw std::runtime_error("Invalid input");
    return size;
}

static Tensor<KeyT> read_tensor() {
    std::size_t batches = read_sizes();
    std::size_t channels = read_sizes();
    std::size_t height = read_sizes();
    std::size_t width = read_sizes();

    Tensor<KeyT> tensor(batches, channels, height, width);

    for (std::size_t batch = 0; batch < batches; ++batch)
        for (std::size_t channel = 0; channel < channels; ++channel)
            for (std::size_t i = 0; i < height; ++i)
                for (std::size_t j = 0; j < width; ++j) {
                    std::cin >> tensor[batch, channel, i, j];
                    if (!std::cin.good()) {
                        throw std::runtime_error("Invalid tensor input");
                    }
                }

    return tensor;
}

static void load_data_once() {
    if (!data_loaded) {
        input = read_tensor();
        weight = read_tensor();
        data_loaded = true;
    }
}

static void BM_Naive_MatMul(benchmark::State& state) {
    load_data_once();
    auto input_node = std::make_shared<InputData<KeyT>>(input);

    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<Naive_MatMulOperation<KeyT>>(input_node, weight));

    for (auto _ : state) {
        benchmark::DoNotOptimize(nn.infer());
    benchmark::ClobberMemory();
    }
}

static void BM_CacheFriendly_MatMul(benchmark::State& state) {
    load_data_once();
    auto input_node = std::make_shared<InputData<KeyT>>(input);

    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<CacheFriendly_MatMulOperation<KeyT>>(input_node, weight));

    for (auto _ : state) {
        benchmark::DoNotOptimize(nn.infer());
    benchmark::ClobberMemory();
    }
}

static void BM_CacheFriendly_Tiling_MatMul(benchmark::State& state) {
    load_data_once();
    auto input_node = std::make_shared<InputData<KeyT>>(input);

    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<CacheFriendly_Tiling_MatMulOperation<KeyT>>(input_node, weight));

    for (auto _ : state) {
        benchmark::DoNotOptimize(nn.infer());
    benchmark::ClobberMemory();
    }
}

static void BM_Optimized_MatMul(benchmark::State& state) {
    load_data_once();
    auto input_node = std::make_shared<InputData<KeyT>>(input);

    network::NeuralNetwork<KeyT> nn;
    nn.addOp(std::make_shared<Optimized_MatMulOperation<KeyT>>(input_node, weight));

    for (auto _ : state) {
        benchmark::DoNotOptimize(nn.infer());
    benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Naive_MatMul)->Iterations(1);
BENCHMARK(BM_CacheFriendly_MatMul)->Iterations(1);
BENCHMARK(BM_CacheFriendly_Tiling_MatMul)->Iterations(1);
BENCHMARK(BM_Optimized_MatMul)->Iterations(1);

BENCHMARK_MAIN();
