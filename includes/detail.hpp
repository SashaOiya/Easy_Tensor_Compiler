#pragma once

#include <vector>
#include "tensor.hpp"

namespace detail {

template <typename KeyT>  // integer
class INode {
   public:
    virtual Tensor<KeyT> evaluate() const = 0;  // Вычисление результата узла
    virtual ~INode() = default;
};

}; // namespace detail


template <typename KeyT>  // integer
class InputData final: public detail::INode<KeyT> {
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

}; 
