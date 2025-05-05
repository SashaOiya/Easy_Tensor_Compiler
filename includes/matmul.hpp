#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include "detail.hpp"

template <typename KeyT>
class Naive_MatMulOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;

   public:
    Naive_MatMulOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        const Tensor<KeyT>& lhs_tensor = node_->evaluate();
        const Tensor<KeyT>& rhs_tensor = width_;

        if (lhs_tensor.num_elements() != rhs_tensor.num_elements()) {
            throw std::invalid_argument("Tensor size mismatch in MatMulOperation");
        }

        Tensor<KeyT> result(lhs_tensor.batch(), lhs_tensor.channels(), lhs_tensor.height(),
                            rhs_tensor.width());

        for (std::size_t i = 0; i < lhs_tensor.num_matrices(); ++i) {
            result.data()[i] = lhs_tensor.data()[i] * rhs_tensor.data()[i];
        }

        return result;
    }

    void setArgs(const std::vector<InputData<KeyT>*>& args) override {
        if (!args.empty()) node_.reset(args[0]);
    }

    const std::vector<InputData<KeyT>*>& getArgs() const override {
        static std::vector<InputData<KeyT>*> args;
        args = {node_.get()};
        return args;
    }
};

template <typename KeyT>
class CacheFriendly_MatMulOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;

   public:
    CacheFriendly_MatMulOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        Tensor<KeyT> lhs_tensor = node_->evaluate();
        Tensor<KeyT> rhs_tensor = width_;

        const std::size_t M = lhs_tensor.height();  
        const std::size_t K = lhs_tensor.width();   
        const std::size_t N = rhs_tensor.width();  

        if (K != rhs_tensor.height()) {
            throw std::invalid_argument("MatMulOperation: Dimension mismatch (K != B.height).");
        }

        Tensor<KeyT> result(lhs_tensor.batch(), lhs_tensor.channels(), M, N);

        for (std::size_t batch = 0; batch < lhs_tensor.batch(); ++batch) {
            for (std::size_t ch = 0; ch < lhs_tensor.channels(); ++ch) {
                const auto& A = lhs_tensor[batch, ch];
                auto& B = rhs_tensor[batch, ch];
                B.transpose();  
                auto& C = result[batch, ch];

                for (std::size_t i = 0; i < M; ++i) {
                    for (std::size_t j = 0; j < N; ++j) {
                        KeyT sum = 0;
                        for (std::size_t k = 0; k < K; ++k) {
                            sum += A[i, k] * B[j, k];  
                        }
                        C[i, j] = sum;
                    }
                }
            }
        }

        return result;
    }

    void setArgs(const std::vector<InputData<KeyT>*>& args) override {
        if (!args.empty()) node_.reset(args[0]);
    }

    const std::vector<InputData<KeyT>*>& getArgs() const override {
        static std::vector<InputData<KeyT>*> args;
        args = {node_.get()};
        return args;
    }
};

template <typename KeyT>
class CacheFriendly_Tiling_MatMulOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;

   public:
   CacheFriendly_Tiling_MatMulOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        Tensor<KeyT> lhs_tensor = node_->evaluate();
        Tensor<KeyT> rhs_tensor = width_;

        const std::size_t M = lhs_tensor.height();
        const std::size_t K = lhs_tensor.width();
        const std::size_t N = rhs_tensor.width();

        if (K != rhs_tensor.height()) {
            throw std::invalid_argument("MatMulOperation: Dimension mismatch (K != B.height).");
        }

        Tensor<KeyT> result(lhs_tensor.batch(), lhs_tensor.channels(), M, N);
        constexpr int TILE_SIZE = 32;

        for (std::size_t batch = 0; batch < lhs_tensor.batch(); ++batch) {
            for (std::size_t ch = 0; ch < lhs_tensor.channels(); ++ch) {
                const auto& A = lhs_tensor[batch, ch];
                auto& B = rhs_tensor[batch, ch];
                auto& C = result[batch, ch];

                B.transpose();

                for (std::size_t i0 = 0; i0 < M; i0 += TILE_SIZE) {
                    const std::size_t i_max = std::min(i0 + TILE_SIZE, M);
                    for (std::size_t j0 = 0; j0 < N; j0 += TILE_SIZE) {
                        const std::size_t j_max = std::min(j0 + TILE_SIZE, N);
                        for (std::size_t k0 = 0; k0 < K; k0 += TILE_SIZE) {
                            const std::size_t k_max = std::min(k0 + TILE_SIZE, K);

                            for (std::size_t i = i0; i < i_max; ++i) {
                                for (std::size_t j = j0; j < j_max; ++j) {
                                    KeyT sum = 0;
                                        for (std::size_t k = k0; k < k_max; ++k)
                                            sum += A[i, k] * B[j, k];

                                    C[i, j] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    void setArgs(const std::vector<InputData<KeyT>*>& args) override {
        if (!args.empty()) node_.reset(args[0]);
    }

    const std::vector<InputData<KeyT>*>& getArgs() const override {
        static std::vector<InputData<KeyT>*> args;
        args = {node_.get()};
        return args;
    }
};

template <typename KeyT>
class Optimized_MatMulOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;

   public:
    Optimized_MatMulOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        Tensor<KeyT> lhs_tensor = node_->evaluate();
        Tensor<KeyT> rhs_tensor = width_;

        const std::size_t M = lhs_tensor.height();
        const std::size_t K = lhs_tensor.width();
        const std::size_t N = rhs_tensor.width();

        if (K != rhs_tensor.height()) {
            throw std::invalid_argument("MatMulOperation: Dimension mismatch (K != B.height).");
        }

        Tensor<KeyT> result(lhs_tensor.batch(), lhs_tensor.channels(), M, N);
        constexpr int TILE_SIZE = 32;

        for (std::size_t batch = 0; batch < lhs_tensor.batch(); ++batch) {
            for (std::size_t ch = 0; ch < lhs_tensor.channels(); ++ch) {
                const auto& A = lhs_tensor[batch, ch];
                auto& B = rhs_tensor[batch, ch];
                auto& C = result[batch, ch];

                B.transpose();

                for (std::size_t i0 = 0; i0 < M; i0 += TILE_SIZE) {
                    const std::size_t i_max = std::min(i0 + TILE_SIZE, M);
                    for (std::size_t j0 = 0; j0 < N; j0 += TILE_SIZE) {
                        const std::size_t j_max = std::min(j0 + TILE_SIZE, N);
                        for (std::size_t k0 = 0; k0 < K; k0 += TILE_SIZE) {
                            const std::size_t k_max = std::min(k0 + TILE_SIZE, K);

                            for (std::size_t i = i0; i < i_max; ++i) {
                                for (std::size_t j = j0; j < j_max; ++j) {
                                    KeyT sum = 0;

                                    if constexpr (std::is_same_v<KeyT, float>) {
                                        __m256 vec_sum = _mm256_setzero_ps();
                                        std::size_t k = k0;
                                        for (; k + 7 < k_max; k += 8) {
                                            __m256 vec_a = _mm256_loadu_ps(&A[i][k]);
                                            __m256 vec_b = _mm256_loadu_ps(&B[j, k]);
                                            vec_sum = _mm256_fmadd_ps(vec_a, vec_b, vec_sum);
                                        }
                                        alignas(32) float temp[8];
                                        _mm256_store_ps(temp, vec_sum);
                                        for (int t = 0; t < 8; ++t) sum += temp[t];

                                        for (; k < k_max; ++k) sum += A[i][k] * B[j, k];
                                    } else {
                                        for (std::size_t k = k0; k < k_max; ++k)
                                            sum += A[i, k] * B[j, k];
                                    }

                                    C[i, j] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    void setArgs(const std::vector<InputData<KeyT>*>& args) override {
        if (!args.empty()) node_.reset(args[0]);
    }

    const std::vector<InputData<KeyT>*>& getArgs() const override {
        static std::vector<InputData<KeyT>*> args;
        args = {node_.get()};
        return args;
    }
};