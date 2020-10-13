#ifndef FPN_H
#define FPN_H
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>



class FPNImpl : public torch::nn::Module {
    public:
        FPNImpl();
        FPNImpl(const std::vector<int> pyramid_levels, int out_channels,
            const std::vector<int> in_channels);
        std::vector<torch::Tensor> forward(std::vector<torch::Tensor> backbone_features);
    private:
    // all necessary ops declaration here
    //can use depth wise separable convolutions as well
    torch::nn::Conv2d p6{nullptr};
    torch::nn::Conv2d p7{nullptr};
    torch::nn::Conv2d lat3{nullptr};
    torch::nn::Conv2d lat4{nullptr};
    torch::nn::Conv2d lat5{nullptr};
    torch::nn::Conv2d out3{nullptr};
    torch::nn::Conv2d out4{nullptr};
    torch::nn::Conv2d out5{nullptr};
    torch::nn::ConvTranspose2d p4_up{nullptr};
    torch::nn::ConvTranspose2d p5_up{nullptr};
    std::vector<torch::Tensor> feat_pyramids;
};

TORCH_MODULE(FPN);

#endif

