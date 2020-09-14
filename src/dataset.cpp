#include "dataset.h"

using namespace cv;


  torch::data::Example<> CustomDataset::get(size_t index)
  {

    std::string file_location = std::get<0>(csv_[index]);
    int64_t label = std::get<1>(csv_[index]);

    // Load image with OpenCV.
    cv::Mat img = cv::imread(file_location);
    if (img.empty())
    {
      std::cout << "Could not read the image: " << file_location << std::endl;
    }

    bool isChar = (img.type() & 0xF) < 2;
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte).to(torch::kFloat).clone();
    img_tensor = img_tensor.permute({2, 0, 1}); // convert to CxHxW

    torch::Tensor label_tensor = torch::full({1}, label).squeeze().to(torch::kInt64);

    return {img_tensor, label_tensor};
  };
