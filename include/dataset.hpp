#pragma once
#ifndef DATASET_HPP
#define DATASET_HPP
#include <torch/data.h>
#include <torch/data/dataloader_options.h>
#include <torch/torch.h>

namespace LibTorchExample {
template <torch::Dtype dtype = torch::kFloat32, int64_t size_length = 10000,
          auto device = torch::kCPU>
class MyDataset : public torch::data::Dataset<MyDataset<>> {
private:
  torch::Dtype _dtype = dtype;
  int64_t _size = size_length;
  torch::DeviceType _device = device;

public:
  using data_label = torch::data::Example<>;
  data_label get(size_t index) override {
    auto label =
        torch::tensor(static_cast<int64_t>(index),
                      torch::TensorOptions().dtype(dtype).device(device));
    auto data = (torch::pow(label, 2) + 3 * index + 2) + torch::rand({1});
    // data = data.reshape({1, 1});
    // label = label.reshape({1, 1});
    return {data, label};
  }
  torch::optional<size_t> size() const override {
    return torch::tensor(
               _size,
               torch::TensorOptions().dtype(torch::kInt64).device(_device))
        .template item<int64_t>();
  }
};
} // namespace LibTorchExample
#endif