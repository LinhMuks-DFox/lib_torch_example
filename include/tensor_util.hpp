#pragma once
#include <string>
#include <torch/nn/module.h>
#include <vector>
#ifndef TENSOR_UTIL_HPP
#define TENSOR_UTIL_HPP
#include <fmt/format.h>
#include <sstream>
#include <torch/torch.h>

namespace LibTorchExample {

    namespace fmt_print {
        inline auto tensor2stdstring(const torch::Tensor& tensor) -> std::string {
            std::stringstream ss;
            ss << "Tensor(\n\t" << tensor << "\n";
            ss << "\tShape(" << tensor.sizes() << ")\n";
            ss << "\tDtype(" << tensor.dtype() << ")\n";
            ss << "\tDevice(" << tensor.device() << ")\n";
            ss << ")";
            return ss.str();
        }
        inline auto module2stdstring(torch::nn::Module& module) -> std::string {
            std::stringstream ss;
            ss << module << "\n";
            return ss.str();
        }
        inline auto print_tensor(const torch::Tensor& tensor) -> void {
        fmt::print("{}\n", fmt_print::tensor2stdstring(tensor));
        }
    };

    inline auto parameter_info_of(torch::nn::Module& module) -> std::string {
        for (const auto& pair : module.named_parameters()) {
            fmt::print("Parameter: {}\n", pair.key());
            fmt_print::print_tensor(pair.value());
        }
        return "Done";
    }
    
}// namespace LibTorchExample

#endif