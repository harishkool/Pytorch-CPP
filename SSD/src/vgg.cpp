#include "vgg.h"
#include <torch/torch.h>
#include <iostream>

VGGImpl::VGGImpl(){
    std::vector<torch::nn::Sequential> vggnet;
    // vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                //   512, 512, 512]
    vgg_conv1 = register_module("vgg_conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_conv2 = register_module("vgg_conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).
                    padding(0).stride(1).dilation(1)));                        

    vgg_maxpool1 = register_module("vgg_maxpool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2)));

    vgg_conv3 = register_module("vgg_conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_conv4 = register_module("vgg_conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_maxpool2 = register_module("vgg_maxpool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2)));

    vgg_conv5 = register_module("vgg_conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_conv6 = register_module("vgg_conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_maxpool3 = register_module("vgg_maxpool3", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2)));

    vgg_conv7 = register_module("vgg_conv7", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_conv8 = register_module("vgg_conv8", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_maxpool4 = register_module("vgg_maxpool4", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride(2)));                        

    vgg_conv9 = register_module("vgg_conv9", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).
                    padding(0).stride(1).dilation(1)));

    vgg_conv10 = register_module("vgg_conv10", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).
                    padding(0).stride(1).dilation(1)));

};

torch::Tensor VGGImpl::forward(torch::Tensor x){
    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv1(x)));
    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv2(x)));
    auto x = vgg_maxpool1(x);

    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv3(x)));
    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv4(x)));
    auto x = vgg_maxpool2(x);

    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv5(x)));
    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv6(x)));
    auto x = vgg_maxpool3(x);

    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv7(x)));
    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv8(x)));
    auto x = vgg_maxpool4(x);

    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv9(x)));
    auto x = torch::nn::BatchNorm2d(torch::nn::ReLU(vgg_conv10(x)));

    return x;
};