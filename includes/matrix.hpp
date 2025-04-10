#pragma once

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
class Matrix final : private std::vector<T> {
   private:
    std::size_t rows_;
    std::size_t cols_;

   public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;
    using reverse_iterator = typename std::vector<value_type>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<value_type>::const_reverse_iterator;
    using iterator_category = typename std::iterator_traits<iterator>::iterator_category;

    using std::vector<value_type>::data;

    Matrix(size_type n_rows, size_type n_cols)
        : std::vector<T>(n_rows * n_cols), rows_(n_rows), cols_(n_cols) {}

    Matrix(size_type n_rows, size_type n_cols, std::initializer_list<value_type> l)
        : rows_(n_rows), cols_(n_cols) {
        if (l.size() != rows_ * cols_) {
            throw std::invalid_argument("Incorrect initializer list size");
        }
        this->assign(l);
    }

    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    Matrix(Matrix&& rhs) noexcept
        : std::vector<value_type>(std::move(rhs)),
          rows_(std::exchange(rhs.rows_, 0)),
          cols_(std::exchange(rhs.cols_, 0)) {}

    Matrix& operator=(Matrix&& rhs) noexcept {
        std::vector<T>::operator=(std::move(rhs));
        rows_ = std::exchange(rhs.rows_, 0);
        cols_ = std::exchange(rhs.cols_, 0);
        return *this;
    }

    ~Matrix() = default;

    constexpr value_type& operator[](size_type row, size_type col) {
        if (!(row < rows_ && col < cols_)) {
            throw std::invalid_argument("Invalid matrix access");
        }
        return data()[row * cols_ + col];
    }

    constexpr const value_type& operator[](size_type row, size_type col) const {
        if (!(row < rows_ && col < cols_)) {
            throw std::invalid_argument("Invalid matrix access");
        }
        return data()[row * cols_ + col];
    }

    //-------------------------------
    size_type n_cols() const noexcept { return cols_; }
    size_type n_rows() const noexcept { return rows_; }
    using std::vector<value_type>::size;

    using std::vector<value_type>::begin;
    using std::vector<value_type>::end;
    using std::vector<value_type>::cbegin;
    using std::vector<value_type>::cend;
    using std::vector<value_type>::rbegin;
    using std::vector<value_type>::rend;
    using std::vector<value_type>::crbegin;
    using std::vector<value_type>::crend;

    bool equal(const Matrix& other) const { return data() == other.data(); }
};

template <typename T>
Matrix<T> multiply(Matrix<T>& A, Matrix<T>& rhs) {
    const std::size_t rhs_cols = rhs.n_cols();
    const std::size_t rhs_rows = rhs.n_rows();
    const std::size_t cols = A.n_cols();
    const std::size_t rows = A.n_rows();

    if (cols != rhs_rows) {
        throw std::invalid_argument("Invalid matrix size");
    }

    Matrix<T> mul_matrix(rows, rhs_cols);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < rhs_cols; ++j) {
            for (std::size_t k = 0; k < cols; ++k) {
                mul_matrix[i, j] += A[i, k] * rhs[k, j];
            }
        }
    }

    return mul_matrix;
}