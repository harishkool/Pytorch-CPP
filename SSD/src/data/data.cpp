#include "data.h"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <iostream>
#include <ATen/ATen.h>
#include <opencv2/opencv.hpp>
#include "json.hpp"

COCODataset::COCODataset(std::string json_pth, std::string img_pth){
    images = getimages(json_pth, img_pth);
    bbx = getbbxs(json_pth);
    classes = getclss(json_pth);
};

std::vector<torch::Tensor> COCODataset::getimages(std::string json_pth, std::string img_pth){
    

};

std::vector<torch::Tensor> COCODataset::getbbxs(std::string json_pth){


};

std::vector<torch::Tensor> COCODataset::getclss(std::string json_pth){


};