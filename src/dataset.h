#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "csv.h"

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
  std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;

public:
  CustomDataset(std::string &file_names_csv)
      // Load csv file with file locations and labels.
      : csv_(ReadCsv(file_names_csv)){

        };

  torch::data::Example<> get(size_t index) override;

  torch::optional<size_t> size() const override
  {

    return csv_.size();
  };
};
