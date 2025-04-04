#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

namespace etc {

// Класс Tensor для хранения данных в формате NCHW
class Tensor {
public:
    // Конструктор: принимает размеры (N, C, H, W) для NCHW формата
    Tensor(int batch = 1, int channels = 1, int height = 1, int width = 1)
        : N_(batch), C_(channels), H_(height), W_(width),
          data_(batch * channels * height * width, 0.0f) {}

    // Доступ к элементу по индексу (N, C, H, W)
    float& operator()(int n, int c, int h, int w) {
        return data_[(n * C_ + c) * H_ * W_ + h * W_ + w];
    }

    // Доступ к элементу по индексу (N, C, H, W) (константный)
    float operator()(int n, int c, int h, int w) const {
        return data_[(n * C_ + c) * H_ * W_ + h * W_ + w];
    }

    // Размеры
    int batch() const { return N_; }
    int channels() const { return C_; }
    int height() const { return H_; }
    int width() const { return W_; }

    // Печать тензора
    void print() const {
        for (int n = 0; n < N_; ++n) {
            std::cout << "Batch " << n << ":\n";
            for (int c = 0; c < C_; ++c) {
                std::cout << " Channel " << c << ":\n";
                for (int h = 0; h < H_; ++h) {
                    for (int w = 0; w < W_; ++w)
                        std::cout << (*this)(n, c, h, w) << " ";
                    std::cout << "\n";
                }
            }
        }
    }

private:
    int N_, C_, H_, W_;   // Размеры (Batch, Channels, Height, Width)
    std::vector<float> data_;  // Данные тензора
};

// Интерфейс узла графа
class INode {
public:
    virtual Tensor evaluate() const = 0;  // Вычисление результата узла
    virtual ~INode() = default;
};

// Интерфейс операции
class IOperation : public INode {
public:
    virtual void setArgs(const std::vector<INode*>& args) = 0;  // Установка аргументов
    virtual const std::vector<INode*>& getArgs() const = 0;     // Получение аргументов
};

// Входные данные — это тензор, обёрнутый в узел
class InputData : public INode {
public:
    explicit InputData(const Tensor& tensor) : tensor_(tensor) {}
    Tensor evaluate() const override { return tensor_; }  // Просто возвращает свой тензор

private:
    Tensor tensor_;
};

// Базовый класс для бинарных операций
class BinaryOperation : public IOperation {
public:
    BinaryOperation(std::shared_ptr<INode> lhs, const Tensor& rhs)
        : lhs_(std::move(lhs)), rhs_(rhs) {}

protected:
    std::shared_ptr<INode> lhs_;  // Входной узел
    Tensor rhs_;                  // Константный вес
};

// Операция сложения
class ScalarAddOperation : public BinaryOperation {
public:
    ScalarAddOperation(std::shared_ptr<INode> lhs, const Tensor& rhs)
        : BinaryOperation(std::move(lhs), rhs) {}

    Tensor evaluate() const override {
        Tensor left = lhs_->evaluate();
        assert(left.batch() == rhs_.batch() && left.channels() == rhs_.channels() && 
               left.height() == rhs_.height() && left.width() == rhs_.width());
        Tensor result(left.batch(), left.channels(), left.height(), left.width());
        for (int n = 0; n < left.batch(); ++n)
            for (int c = 0; c < left.channels(); ++c)
                for (int h = 0; h < left.height(); ++h)
                    for (int w = 0; w < left.width(); ++w)
                        result(n, c, h, w) = left(n, c, h, w) + rhs_(n, c, h, w);
        return result;
    }

    void setArgs(const std::vector<INode*>& args) override {
        if (!args.empty()) lhs_.reset(args[0]);
    }

    const std::vector<INode*>& getArgs() const override {
        static std::vector<INode*> args;
        args = { lhs_.get() };
        return args;
    }
};

// Класс нейросети: управляет операциями
class NeuralNetwork {
public:
    // Добавить операцию в граф
    std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op) {
        ops_.push_back(op);
        return op;
    }

    // Вычислить результат всей сети
    Tensor infer() {
        assert(!ops_.empty());
        return ops_.back()->evaluate();
    }

private:
    std::vector<std::shared_ptr<IOperation>> ops_;  // Операции сети
};

}  // namespace etc

int main() {
    using namespace etc;

    // Создаём входной тензор NCHW (1, 1, 2, 2)
    Tensor input(1, 1, 2, 2);
    input(0, 0, 0, 0) = 1; input(0, 0, 0, 1) = 2;
    input(0, 0, 1, 0) = 3; input(0, 0, 1, 1) = 4;

    // Задаём веса (тоже NCHW)
    Tensor weight(1, 1, 2, 2);
    weight(0, 0, 0, 0) = 10; weight(0, 0, 0, 1) = 20;
    weight(0, 0, 1, 0) = 30; weight(0, 0, 1, 1) = 40;

    // Заворачиваем входной тензор в InputData узел
    auto input_node = std::make_shared<InputData>(input);

    // Создаём сеть и добавляем операцию сложения
    NeuralNetwork nn;
    nn.addOp(std::make_shared<ScalarAddOperation>(input_node, weight));

    // Запускаем вычисление сети и выводим результат
    Tensor output = nn.infer();
    output.print();  // Печатаем результат

    return 0;
}
