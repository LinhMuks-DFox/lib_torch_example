#pragma once
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#ifndef MODEL_HPP
#define MODEL_HPP

#include <torch/torch.h>

namespace LibTorchExample {
    class MyModel : public torch::nn::Module {
    private:
        torch::nn::Sequential convblock1   { nullptr };
        torch::nn::Sequential convblock2   { nullptr };
        torch::nn::Flatten    flatten      { nullptr };
        torch::nn::Sequential ffn          { nullptr };
    public:
        explicit MyModel() {
            this->convblock1 = register_module(
                    "convblock1",
                    torch::nn::Sequential(
                            torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(
                                            1)),
                            torch::nn::ReLU(),
                            torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(
                                            1))));
            this->convblock2 = register_module(
                    "convblock2",
                    torch::nn::Sequential(
                            torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(
                                            1)),
                            torch::nn::ReLU(),
                            torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(
                                            1))));
            this->flatten = register_module("flatten", torch::nn::Flatten());
            this->ffn = register_module(
                    "ffn",
                    torch::nn::Sequential(
                            torch::nn::Linear(256 * 28 * 28, 1024),
                            torch::nn::ReLU(),
                            torch::nn::Linear(1024, 10)));
        }
        MyModel(const MyModel &other) = delete;
        MyModel &operator=(const MyModel &other) = delete;
        MyModel(MyModel &&other) {
            this->convblock1 = std::move(other.convblock1);
            this->convblock2 = std::move(other.convblock2);
            this->flatten = std::move(other.flatten);
            this->ffn = std::move(other.ffn);
        }
        MyModel &operator=(MyModel &&other) {
            if (this != &other) {
                this->convblock1 = std::move(other.convblock1);
                this->convblock2 = std::move(other.convblock2);
                this->flatten = std::move(other.flatten);
                this->ffn = std::move(other.ffn);
            }
            return *this;
        }
        torch::Tensor forward(const torch::Tensor &x) {
            auto out = this->convblock1->forward(x);
            out = this->convblock2->forward(out);
            out = this->flatten->forward(out);
            out = this->ffn->forward(out);
            return x;
        }
        torch::nn::Sequential get_convblock1() { return this->convblock1; }
        torch::nn::Sequential get_convblock2() { return this->convblock2; }
        torch::nn::Sequential get_ffn() { return this->ffn; }
    };
}

#endif