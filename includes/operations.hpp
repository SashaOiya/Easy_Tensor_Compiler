#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <immintrin.h>

#include "detail.hpp"
#include "matmul.hpp"

template <typename KeyT>
class ScalarAddOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;
    using size_type = std::size_t;

   public:
    ScalarAddOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        Tensor<KeyT> left = node_->evaluate();

        assert(left.batch() == width_.batch() && left.channels() == width_.channels() &&
               left.height() == width_.height() && left.width() == width_.width());
        
        Tensor<KeyT> result(left.batch(), left.channels(), left.height(), left.width());
        for (size_type n = 0; n < left.batch(); ++n)  // std::size_t
            for (size_type c = 0; c < left.channels(); ++c)
                for (size_type h = 0; h < left.height(); ++h)
                    for (size_type w = 0; w < left.width(); ++w)
                        result[n, c, h, w] = left[n, c, h, w] + width_[n, c, h, w];
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
class ScalarSubOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;
    using size_type = std::size_t;

   public:
    ScalarSubOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        Tensor<KeyT> left = node_->evaluate();

        assert(left.batch() == width_.batch() && left.channels() == width_.channels() &&
               left.height() == width_.height() && left.width() == width_.width());

        Tensor<KeyT> result(left.batch(), left.channels(), left.height(), left.width());
        for (size_type n = 0; n < left.batch(); ++n)  // std::size_t
            for (size_type c = 0; c < left.channels(); ++c)
                for (size_type h = 0; h < left.height(); ++h)
                    for (size_type w = 0; w < left.width(); ++w)
                        result[n, c, h, w] = left[n, c, h, w] - width_[n, c, h, w];
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
class ScalarMulOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;
    using size_type = std::size_t;

   public:
    ScalarMulOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        Tensor<KeyT> left = node_->evaluate();

        assert(left.batch() == width_.batch() && left.channels() == width_.channels() &&
               left.height() == width_.height() && left.width() == width_.width());

        Tensor<KeyT> result(left.batch(), left.channels(), left.height(), left.width());
        for (size_type n = 0; n < left.batch(); ++n)  // std::size_t
            for (size_type c = 0; c < left.channels(); ++c)
                for (size_type h = 0; h < left.height(); ++h)
                    for (size_type w = 0; w < left.width(); ++w)
                        result[n, c, h, w] = left[n, c, h, w] * width_[n, c, h, w];
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
class ConvolOperation : public detail::BinaryOperation<KeyT> {
   public:
    using TensorT = Tensor<KeyT>;
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;

    ConvolOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    TensorT evaluate() const override {
        const TensorT& input_tensor = node_->evaluate();
        const TensorT& kernel_tensor = width_;

        if (input_tensor.channels() != kernel_tensor.channels()) {
            throw std::invalid_argument("Convolution: input and kernel channels must match.");
        }

        const auto batch = input_tensor.batch();
        const auto out_channels = kernel_tensor.batch();  // кол-во фильтров
        const auto kernel_height = kernel_tensor.height();
        const auto kernel_width = kernel_tensor.width();
        const auto out_height = input_tensor.height() - kernel_height + 1;
        const auto out_width = input_tensor.width() - kernel_width + 1;

        TensorT result(batch, out_channels, out_height, out_width);

        for (std::size_t n = 0; n < batch; ++n) {
            for (std::size_t oc = 0; oc < out_channels; ++oc) {
                for (std::size_t h = 0; h < out_height; ++h) {
                    for (std::size_t w = 0; w < out_width; ++w) {
                        KeyT sum = 0;
                        for (std::size_t c = 0; c < input_tensor.channels(); ++c) {
                            for (std::size_t kh = 0; kh < kernel_height; ++kh) {
                                for (std::size_t kw = 0; kw < kernel_width; ++kw) {
                                    sum += input_tensor[n, c, h + kh, w + kw] *
                                           kernel_tensor[oc, c, kh, kw];
                                }
                            }
                        }
                        result[n, oc, h, w] = sum;
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

template <typename T>
class ReLUOperation : public detail::UnaryOperation<T> {
    using detail::UnaryOperation<T>::arg_;

   public:
    ReLUOperation(std::shared_ptr<InputData<T>> lhs) : detail::UnaryOperation<T>(std::move(lhs)) {}

    Tensor<T> evaluate() const override {
        Tensor<T> input_tensor = arg_->evaluate();
        Tensor<T> result = input_tensor;

        for (std::size_t i = 0; i < result.num_matrices(); ++i) {
            auto& mat = result.data()[i];
            for (std::size_t r = 0; r < mat.n_rows(); ++r) {
                for (std::size_t c = 0; c < mat.n_cols(); ++c) {
                    mat[r, c] = std::max<T>(0, mat[r, c]);
                }
            }
        }

        return result;
    }

    void setArgs(const std::vector<InputData<T>*>& args) override {
        if (!args.empty()) arg_.reset(args[0]);
    }

    const std::vector<InputData<T>*>& getArgs() const override {
        static std::vector<InputData<T>*> args;
        args = {arg_.get()};
        return args;
    }
};

template <typename T>
class SoftmaxOperation : public detail::UnaryOperation<T> {
    using detail::UnaryOperation<T>::arg_;

   public:
    SoftmaxOperation(std::shared_ptr<InputData<T>> lhs)
        : detail::UnaryOperation<T>(std::move(lhs)) {}

    Tensor<T> evaluate() const override {
        Tensor<T> input_tensor = arg_->evaluate();
        Tensor<T> result = input_tensor;

        for (std::size_t i = 0; i < result.num_matrices(); ++i) {
            auto& mat = result.data()[i];
            for (std::size_t r = 0; r < mat.n_rows(); ++r) {
                T sum_exp = 0;
                std::vector<T> exp_vals(mat.n_cols());

                for (std::size_t c = 0; c < mat.n_cols(); ++c) {
                    exp_vals[c] = std::exp(mat[r, c]);
                    sum_exp += exp_vals[c];
                }

                for (std::size_t c = 0; c < mat.n_cols(); ++c) {
                    mat[r, c] = exp_vals[c] / sum_exp;
                }
            }
        }

        return result;
    }

    void setArgs(const std::vector<InputData<T>*>& args) override {
        if (!args.empty()) arg_.reset(args[0]);
    }

    const std::vector<InputData<T>*>& getArgs() const override {
        static std::vector<InputData<T>*> args;
        args = {arg_.get()};
        return args;
    }
};
