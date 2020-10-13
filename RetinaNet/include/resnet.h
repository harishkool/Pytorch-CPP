#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>
#include <iostream>
#include <string>

class BasicblockImpl : public torch::nn::Module{
    public:
        BasicblockImpl();
        BasicblockImpl(int inplanes, int outplanes, 
                int stride, int base_width, int dilation, std::string norm_type);
        torch::Tensor forward(torch::Tensor x);
    private:
        // All operations basic block needed
        torch::nn::Conv2d conv_1{nullptr};
        torch::nn::Conv2d conv_2{nullptr};
        torch::nn::BatchNorm2d batchnorm1{nullptr};
        torch::nn::BatchNorm2d batchnorm2{nullptr};
};

TORCH_MODULE(Basicblock);

class BottleneckblockImpl: public torch::nn::Module{
    public:
        BottleneckblockImpl();
        BottleneckblockImpl(int inplanes, int outplanes, 
                int stride, int groups, int base_width, int dilation, std::string norm_type);
        torch::Tensor forward(torch::Tensor x);
    private:
        // All bottleneckblock operations needed
        torch::nn::Conv2d conv_1{nullptr};
        torch::nn::Conv2d conv_2{nullptr};
        torch::nn::Conv2d conv_3{nullptr};
        torch::nn::BatchNorm2d batchnorm1{nullptr};
        torch::nn::BatchNorm2d batchnorm2{nullptr};
        torch::nn::BatchNorm2d batchnorm3{nullptr};
};

TORCH_MODULE(Bottleneckblock);

class ResNetImpl : public torch::nn::Module{
    public:
        ResNetImpl();
        ResNetImpl(std::string resnet_arch, std::string norm_type, std::vector<int> arch_layers,
            int input_channels, int groups, int width);
        auto forward(torch::Tensor x);
    private:
        // All operations needed for various ResNet implementations
    torch::nn::Conv2d conv_1{nullptr};
    torch::nn::MaxPool2d maxpool_1{nullptr};
    torch::nn::BatchNorm2d batchnorm1{nullptr};
    torch::nn::Sequential c2{nullptr};
    torch::nn::Sequential c3{nullptr};
    torch::nn::Sequential c4{nullptr};
    torch::nn::Sequential c5{nullptr};
};


TORCH_MODULE(ResNet);

#endif