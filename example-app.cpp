#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;

// Where to find the MNIST dataset.
const char *kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 1;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1;

auto ReadCsv(std::string &location) -> std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>>
{

  std::fstream in(location, std::ios::in);
  std::string line;
  std::string name;
  std::string label;
  std::vector<std::tuple<std::string, int64_t>> csv;

  while (getline(in, line))
  {
    std::stringstream s(line);
    getline(s, name, ',');
    std::cout << name << std::endl;
    getline(s, label, ',');

    csv.push_back(std::make_tuple(name, std::stoi(label)));
  }

  return csv;
}

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
  std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;

public:
  explicit CustomDataset(std::string &file_names_csv)
      // Load csv file with file locations and labels.
      : csv_(ReadCsv(file_names_csv)){

        };

  // Override the get method to load custom data.
  torch::data::Example<> get(size_t index) override
  {

    std::string file_location = std::get<0>(csv_[index]);
    int64_t label = std::get<1>(csv_[index]);

    // Load image with OpenCV.
    std::cout << file_location << std::endl;
    cv::Mat img = cv::imread(file_location);
    if (img.empty())
    {
      std::cout << "Could not read the image: " << file_location << std::endl;
    }

    // Convert the image and label to a tensor.
    // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
    // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
    // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
    bool isChar = (img.type() & 0xF) < 2;
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, isChar ? torch::kChar : torch::kFloat).to(torch::kFloat).clone();
    img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW
    std::cout << img_tensor << std::endl;

    torch::Tensor label_tensor = torch::full({1}, label);

    return {img_tensor, label_tensor};
  };

  // Override the size method to infer the size of the data set.
  torch::optional<size_t> size() const override
  {

    return csv_.size();
  };
};

struct Net : torch::nn::Module
{
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10)
  {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2->forward(x), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};

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
  std::cout << "Training" << std::endl;

  for (torch::data::Example<> &batch : data_loader)
  {
    std::cout << "Training" << std::endl;
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();

    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();
    if (batch_idx++ % kLogInterval == 0)
    {
      std::printf(
          "\rTrain Epoch: %ld [%zd/%zd] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size)
{
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto &batch : data_loader)
  {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{})
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
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
  std::string file_names_csv = "train.csv";
  auto train_dataset = CustomDataset(file_names_csv).map(torch::data::transforms::Stack<>());

  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset,
      kTrainBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
  model.train();
  size_t batch_idx = 0;
  std::cout << "Training" << std::endl;

  size_t epoch = 1;
  for (auto &batch : *train_loader)
  {
    std::cout << "Training" << std::endl;
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();

    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();
    std::printf(
        "\rTrain Epoch: %zd [%zd/%zd] Loss: %.4f",
        epoch,
        batch_idx * batch.data.size(0),
        train_dataset_size,
        loss.template item<float>());
  }

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
  {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
  }
}
