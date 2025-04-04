#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "matrix.hpp"

template <typename T>
class Tensor final {
    std::size_t batch_, channels_, height_, width_;  // Размеры NCHW
    std::vector<Matrix<T>> data_;

   public:
    using size_type = std::size_t;
    using value_type = Matrix<T>;
    using difference_type = std::ptrdiff_t;  // ?
    using reference = value_type&;
    using const_reference = const value_type&;

    Tensor(size_type batch = 1, size_type channels = 1, size_type height = 1, size_type width = 1)
        : batch_(batch),
          channels_(channels),
          height_(height),
          width_(width),
          data_(batch_ * channels_, Matrix<T>(height_, width_)) {}

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&& rhs) noexcept
        : std::vector<value_type>(std::move(rhs)),
          batch_(std::exchange(rhs.batch_, 0)),
          channels_(std::exchange(rhs.channels_, 0)),
          height_(std::exchange(rhs.height_, 0)),
          width_(std::exchange(rhs.width_, 0)) {}

    Tensor& operator=(Tensor&& rhs) noexcept {
        std::vector<value_type>::operator=(std::move(rhs));
        batch_ = std::exchange(rhs.batch_, 0);
        channels_ = std::exchange(rhs.channels_, 0);
        height_ = std::exchange(rhs.height_, 0);
        width_ = std::exchange(rhs.width_, 0);
    }

    ~Tensor() = default;

    //--------------------------------
    constexpr value_type& operator[](size_type n, size_type c) {
        assert(i < batch_ && j < channels_);
        return data_[n * channels_ + c];
    }

    constexpr value_type& operator[](size_type n, size_type c) const {
        assert(i < batch_ && j < channels_);
        return data_[n * channels_ + c];
    }

    constexpr T& operator[](size_type n, size_type c, size_type h, size_type w) {
        assert(i < batch_ && j < channels_);
        return data_[n * channels_ + c][h, w];
    }

    constexpr T& operator[](size_type n, size_type c, size_type h, size_type w) const {
        assert(i < batch_ && j < channels_);
        return data_[n * channels_ + c][h, w];
    }

    int batch() const { return batch_; }
    int channels() const { return channels_; }
    int height() const { return height_; }
    int width() const { return width_; }

    size_type size() const noexcept { return batch_ * channels_ * height_ * width_; }
    constexpr auto& data() noexcept { return data_; }
    constexpr auto& data() const noexcept { return data_; }

    void dump() const {
        for (int n = 0; n < batch_; ++n) {
            std::cout << "Batch " << n << ":\n";
            for (int c = 0; c < channels_; ++c) {
                std::cout << " Channel " << c << ":\n";
                for (int h = 0; h < height_; ++h) {
                    for (int w = 0; w < width_; ++w) std::cout << (*this)[n, c, h, w] << " ";
                    std::cout << "\n";
                }
            }
        }
    }
};