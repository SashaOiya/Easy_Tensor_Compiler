#include <gtest/gtest.h>

#include "tensor.hpp"

TEST(tensor, ctor) {
    Tensor<double> tensor = {2, 2, 2, 2};

    EXPECT_EQ(tensor.batch(), 2);
    EXPECT_EQ(tensor.channels(), 2);
    EXPECT_EQ((tensor[0,0].size()), 4);
    EXPECT_EQ((tensor[0, 1].size()), 4);
    EXPECT_EQ((tensor[1,1].size()), 4);
}

TEST(tensor, ctor_init) {
    Tensor<double> tensor = {2, 2, 2, 2};
    tensor[1, 1, 1, 0] = 4;

    EXPECT_EQ((tensor[1,1,1,0]), 4);
}