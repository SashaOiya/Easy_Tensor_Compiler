#pragma once

#include <cstddef>
#include <memory>

#include "detail.hpp"

namespace detail {

template <typename KeyT>
class BinaryOperation : public IOperation<KeyT> {
   public:
    BinaryOperation(std::shared_ptr<InputData<KeyT>> node, const Tensor<KeyT>& width)
        : node_(std::move(node)), width_(width) {}

   protected:
    std::shared_ptr<InputData<KeyT>> node_;  // Входной узел
    Tensor<KeyT> width_;                     // Константный вес
};

};  // namespace detail

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
class MatMulOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;

   public:
    MatMulOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
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
