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
        RetinaHeadImpl(int in_channels, int anchors_per_cell, 
            int num_convs, float prior, float drop_rate);
        std::vector<torch::Tensor> forward(std::vector<torch::Tensor> feature_pyramids);
};

class RetinaNetImpl: public torch::nn::Module{
    public:
        RetinaNetImpl();
        RetinaNetImpl(const std::string backbone_arch, const std::vector<int> pyramid_levels);
        at::Tensor forward(torch::Tensor x);
    private:
        // All the necessary operations needed for RetinaNet implmentation
        std::shared_ptr<ResNet> backbone;
};

TORCH_MODULE(RetinaNet);

#endif