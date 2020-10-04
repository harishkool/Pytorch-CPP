#ifndef DATA_H
#define DATA_H
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <string>
#include <vector>
#include "json.hpp"

class COCODataset:public torch::data::Dataset<COCODataset>{
    private:
        std::vector<torch::Tensor>images, bbx, classes;

    public:
        COCODataset(std::string json_pth, std::string img_pth);
        std::vector<torch::Tensor> getimages(std::string json_pth, std::string img_pth);
        std::vector<torch::Tensor> getbbxs(std::string json_pth);
        std::vector<torch::Tensor> getclss(std::string json_pth);
        torch::data::Example<> get(size_t indx) override;
        torch::optional<size_t> size() const override;
};


#endif