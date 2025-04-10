#pragma once

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

// Операция сложения
template <typename KeyT>
class ScalarAddOperation : public detail::BinaryOperation<KeyT> {
    using detail::BinaryOperation<KeyT>::node_;
    using detail::BinaryOperation<KeyT>::width_;

   public:
    ScalarAddOperation(std::shared_ptr<InputData<KeyT>> lhs, const Tensor<KeyT>& rhs)
        : detail::BinaryOperation<KeyT>(std::move(lhs), rhs) {}

    Tensor<KeyT> evaluate() const override {
        Tensor<KeyT> left = node_->evaluate();

        assert(left.batch() == width_.batch() && left.channels() == width_.channels() &&
               left.height() == width_.height() && left.width() == width_.width());
               

        Tensor<KeyT> result(left.batch(), left.channels(), left.height(), left.width());
        for (int n = 0; n < left.batch(); ++n)  // std::size_t
            for (int c = 0; c < left.channels(); ++c)
                for (int h = 0; h < left.height(); ++h)
                    for (int w = 0; w < left.width(); ++w)
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