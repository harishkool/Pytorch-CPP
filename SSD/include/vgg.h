#ifndef VGG_H
#define VGG_H

#include<torch/torch.h>

class VGGImpl: public torch::nn::Module{
    public:
        VGGImpl();
        torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::Conv2d vgg_conv1{nullptr};
        torch::nn::Conv2d vgg_conv2{nullptr};
        torch::nn::Conv2d vgg_conv3{nullptr};
        torch::nn::Conv2d vgg_conv4{nullptr};
        torch::nn::Conv2d vgg_conv5{nullptr};
        torch::nn::Conv2d vgg_conv6{nullptr};
        torch::nn::Conv2d vgg_conv7{nullptr};
        torch::nn::Conv2d vgg_conv8{nullptr};
        torch::nn::Conv2d vgg_conv9{nullptr};
        torch::nn::Conv2d vgg_conv10{nullptr};
        torch::nn::MaxPool2d vgg_maxpool1{nullptr};
        torch::nn::MaxPool2d vgg_maxpool2{nullptr};
        torch::nn::MaxPool2d vgg_maxpool3{nullptr};
        torch::nn::MaxPool2d vgg_maxpool4{nullptr};
    };

#endif