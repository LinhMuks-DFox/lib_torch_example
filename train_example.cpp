#include <torch/torch.h>
#include "include/dataset.hpp"
#include "include/tensor_util.hpp"
#include <fmt/format.h>
using LibTorchExample::fmt_print::tensor2stdstring;

auto train() -> int {
    auto model = torch::nn::Sequential(
            torch::nn::Linear(1, 1), torch::nn::ReLU(),
            torch::nn::Linear(1, 1)
            );
    auto dataset = LibTorchExample::MyDataset<torch::kFloat32, 10000, torch::kCPU>();
    auto dataloader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                    std::move(dataset),
                    torch::data::DataLoaderOptions().batch_size(500).workers(2));

    auto optimizer =
            torch::optim::Adam(model->parameters(), torch::optim::AdamOptions().lr(1e-4));
    auto loss_function = torch::nn::MSELoss();

    for (int epoch = 0; epoch < 50; ++epoch) {
        // log epoch loss by using std::vector
        std::vector<torch::Tensor> epoch_loss;
        for (auto &batch : *dataloader) {

            std::vector<torch::Tensor> data_vec, target_vec;
            std::transform(batch.begin(), batch.end(), std::back_inserter(data_vec),
                           [](const auto &example) { return example.data; });
            std::transform(batch.begin(), batch.end(), std::back_inserter(target_vec),
                           [](const auto &example) { return example.target; });

            auto data = torch::stack(data_vec).to(torch::kFloat32);// 确保数据类型正确
            auto label = torch::stack(target_vec).to(torch::kFloat32);// 确保目标类型正确
            label = label.reshape({ -1, 1 });
            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = loss_function(output, label);
            loss.backward();
            optimizer.step();
            epoch_loss.push_back(loss);
        }
        auto epoch_loss_tensor = torch::stack(epoch_loss);
        fmt::print("Epoch: {}\nLoss: {}\n", epoch,
                   tensor2stdstring(epoch_loss_tensor.mean()));
    }
    return 0;
}
