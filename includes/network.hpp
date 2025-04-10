#pragma once

#include "tensor.hpp"
#include "operations.hpp"

namespace network {

template <typename KeyT>  // integer
class NeuralNetwork {
   public:
    std::shared_ptr<detail::IOperation<KeyT>> addOp(std::shared_ptr<detail::IOperation<KeyT>> op) {
        ops_.push_back(op);
        return op;
    }

    // Вычислить результат всей сети
    Tensor<KeyT> infer() {
        assert(!ops_.empty());
        return ops_.back()->evaluate();
    }

   private:
    std::vector<std::shared_ptr<detail::IOperation<KeyT>>> ops_;  // Операции сети
};

};  // namespace network