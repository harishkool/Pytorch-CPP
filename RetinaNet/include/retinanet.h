#ifndef RETINANET_H
#define RETINANET_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include "resnet.h"


class RetinaHeadImpl : public torch::nn::Module{
    public:
        RetinaHeadImpl();
        RetinaHeadImpl(int in_channels, int num_classes, int anchors_per_cell, 
            int num_convs, float prior, float drop_rate);
        std::pair<std::vector<torch::Tensor>,std::vector<torch::Tensor>> forward(std::vector<torch::Tensor> feature_pyramids);
    private:
        torch::nn::Sequential bbox_subnet{nullptr};
        torch::nn::Sequential clss_subnet{nullptr};
        torch::nn::Conv2d bbox_pred{nullptr};
        torch::nn::Conv2d clss_pred{nullptr};
};

TORCH_MODULE(RetinaHead);

class RetinaNetImpl: public torch::nn::Module{
    public:
        RetinaNetImpl();
        RetinaNetImpl(const std::string backbone_arch, const std::vector<int> pyramid_levels, 
                int head_channels, int num_classes, int anchors_per_cell, int num_convs);
        at::Tensor forward(torch::Tensor x);
    private:
        // All the necessary operations needed for RetinaNet implmentation
        std::shared_ptr<ResNet> backbone;
        std::shared_ptr<FPN> fpn_ptr;
        std::shared_ptr<RetinaHead> retinahead_ptr;
        ResNet backbone_cp;
};

TORCH_MODULE(RetinaNet);

#endif