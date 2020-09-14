#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <inttypes.h>

#include "dataset.h"
#include "model.h"

// The batch size for training.
const int64_t kTrainBatchSize = 1;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 2;

template <typename DataLoader>
void train(
    int32_t epoch,
    Net &model,
    torch::Device device,
    DataLoader &data_loader,
    torch::optim::Optimizer &optimizer,
    size_t dataset_size)
{
  model.train();
  size_t batch_idx = 0;

  for (torch::data::Example<> &batch : data_loader)
  {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();
    batch_idx++;
    std::fflush(stdout);
    std::printf(
      "\rTrain Epoch: %" PRId32 " [%zd/%zd] Loss: %.4f",
      epoch,
      batch_idx * batch.data.size(0),
      dataset_size,
      loss.template item<float>());
  }
  std::cout << std::endl << "Epoch ended" << std::endl;
}

int main(int argc, char **argv)
{
  // if ( argc != 2 )
  // {
  //     printf("usage: DisplayImage.out <Image_Path>\n");
  //     return -1;
  // }

  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  }
  else
  {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  Net model;
  model.to(device);
  std::string file_names_csv = "../train.csv";
  auto train_dataset = CustomDataset(file_names_csv).map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                                                    .map(torch::data::transforms::Stack<>());

  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset,
      kTrainBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
  {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
  }
}
