#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "tensor.hpp"

namespace detail {

template <typename KeyT>  // integer
class INode {
   public:
    virtual Tensor<KeyT> evaluate() const = 0;  
    virtual ~INode() = default;
};

};  // namespace detail

template <typename KeyT>  // integer
class InputData final : public detail::INode<KeyT> {
   public:
    explicit InputData(const Tensor<KeyT>& tensor) : tensor_(tensor) {}
    Tensor<KeyT> evaluate() const override { return tensor_; }
    ~InputData() override = default;

   private:
    Tensor<KeyT> tensor_;
};

namespace detail {

template <typename KeyT>
class IOperation : public INode<KeyT> {
   public:
    virtual void setArgs(const std::vector<InputData<KeyT>*>& args) = 0;
    virtual const std::vector<InputData<KeyT>*>& getArgs() const = 0;
};

template <typename KeyT>
class BinaryOperation : public IOperation<KeyT> {
   public:
    BinaryOperation(std::shared_ptr<InputData<KeyT>> node, const Tensor<KeyT>& width)
        : node_(std::move(node)), width_(width) {}

   protected:
    std::shared_ptr<InputData<KeyT>> node_;  // Входной узел
    Tensor<KeyT> width_;                     // Константный вес
};

template <typename KeyT>
class UnaryOperation : public IOperation<KeyT> {
   public:
    explicit UnaryOperation(const std::shared_ptr<InputData<KeyT>>& arg) : arg_(arg) {}

   protected:
    std::shared_ptr<InputData<KeyT>> arg_;
};

};  // namespace detail
