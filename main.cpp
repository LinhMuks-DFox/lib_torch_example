#include "fmt/base.h"
#include "include/dataset.hpp"
#include "include/model.hpp"
#include "include/tensor_util.hpp"
#include <bits/ranges_algo.h>
#include <c10/core/DeviceType.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim/adam.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <torch/types.h>

using LibTorchExample::fmt_print::tensor2stdstring;
using LibTorchExample::fmt_print::module2stdstring;

int main(int argc, char *argv[]) {
    auto model2 = LibTorchExample::MyModel();
    model2.to(torch::DeviceType::CUDA);
    fmt::print("Model: {}\n", module2stdstring(model2));
    return 0;
}
