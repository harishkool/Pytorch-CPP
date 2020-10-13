#include <torch/torch.h>
#include <iostream>
#include "fpn.h"


FPNImpl::FPNImpl(){

}

FPNImpl::FPNImpl(const std::vector<int> pyramid_levels, int out_channels,
            const std::vector<int> in_channels){
                
        // in_channels --> [512, 1024, 2048] --> from resnet or any backbone
        // pyramid_levels --> [3, 4, 5, 6, 7] --> these are the levels used in the paper retinanet
        int sz = in_channels.size();
        p6 = register_module("p6", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels[sz-1], out_channels, 3).padding(1)
                .stride(2)));

        p7 = register_module("p6", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)
                .stride(2)));

        lat3 = register_module("lat3", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels[0], out_channels, 1)
            .stride(1).padding(0)));


        lat4 = register_module("lat4", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels[1], out_channels, 1)
            .stride(1).padding(0)));


        lat5 = register_module("lat5", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels[sz-1], out_channels, 1)
            .stride(1).padding(0)));

        out3 = register_module("out3", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
            .stride(1)));

        out4 = register_module("out4", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
            .stride(1)));

        out5 = register_module("out5", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
            .stride(1)));

        p4_up = register_module("p4_up", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(out_channels, out_channels, 2)
                .stride(2).groups(out_channels)));


        p5_up = register_module("p5_up", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(out_channels, out_channels, 2)
                .stride(2).groups(out_channels)));
        
}


std::vector<torch::Tensor> FPNImpl::forward(std::vector<torch::Tensor> backbone_features){
    // backbone_features --> contains c3, c4, c5 from resnet backbone, corresponding from any other backbone
    int sz = backbone_features.size()-1;
    std::vector<torch::Tensor> pyramid_features;

    torch::Tensor p5_t = lat5(backbone_features[sz]);
    torch::Tensor  p5_up_t = p5_up(p5_t);
    p5_t = out5(p5_t);


    torch::Tensor p4_t = torch::add(lat4(backbone_features[sz-1]), p5_t);
    torch::Tensor  p4_up_t = p4_up(p4_t);
    p4_t = out5(p4_t);

    torch::Tensor p3_t = torch::add(lat3(backbone_features[sz-2]), p4_t);
    p3_t = out3(p3_t);

    torch::Tensor p6_t = p6(backbone_features[sz]);

    torch::Tensor p7_t = torch::relu(p7(p6_t));

    pyramid_features.push_back(p3_t);
    pyramid_features.push_back(p4_t);
    pyramid_features.push_back(p5_t);
    pyramid_features.push_back(p6_t);
    pyramid_features.push_back(p7_t);

    return pyramid_features;
}