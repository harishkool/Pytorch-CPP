#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "resnet.h"


torch::nn::Conv2d conv3x3 (int inplanes, int outplanes,
                            int stride=1, int groups=1, int dilation=1){
            
                    return torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, outplanes, 3).padding(0).
                                stride(stride).dilation(dilation).bias(false));
    }

torch::nn::Conv2d conv1x1 (int inplanes, int outplanes, int stride=1){

                    return torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, outplanes, 1).stride(1).bias(false));
}

// Basic block implementation 
BasicblockImpl::BasicblockImpl(){}
BasicblockImpl::BasicblockImpl(int inplanes, int outplanes, int stride, int basewidth, int dilation, 
            std::string norm_type="batch_norm"){
// perform conv 3 x 3, batchnorm, conv 3 x 3, batchnorm, add first feature to this one
    conv_1 = register_module("conv_1", conv3x3(inplanes, outplanes, stride));
    batchnorm1 = register_module("batchnorm1", torch::nn::BatchNorm2d(outplanes));
    conv_2 = register_module("conv_2", conv3x3(outplanes, outplanes, stride));
    batchnorm2 = register_module("batchnorm2", torch::nn::BatchNorm2d(outplanes));
}

at::Tensor BasicblockImpl::forward(torch::Tensor x){
    torch::Tensor identity = x;
    torch::Tensor out;
    x = torch::relu(batchnorm1->forward(conv_1->forward(x)));
    x = torch::relu(batchnorm2->forward(conv_2->forward(x)));
    x = torch::relu(torch::add(identity, x));
    return x;
}

//Bottleneck block implementation
BottleneckblockImpl::BottleneckblockImpl(){}
BottleneckblockImpl::BottleneckblockImpl(int inplanes, int outplanes, int stride, 
                    int groups, int basewidth, int dilation, std::string norm_type="batch_norm"){
//  
    conv_1 = register_module("conv_1", conv1x1(inplanes, outplanes, stride));
    batchnorm1 = register_module("batchnorm1", torch::nn::BatchNorm2d(outplanes));
    int width = int(outplanes * (basewidth / 64.0)) * groups;
    conv_2 = register_module("conv_2", conv3x3(width, width, stride, groups, dilation));
    batchnorm2 = register_module("batchnorm2", torch::nn::BatchNorm2d(width));
    conv_3 = register_module("conv_3", conv1x1(width, outplanes * 4));
    batchnorm3 = register_module("batchnorm3", torch::nn::BatchNorm2d(outplanes * 4));
                
}
at::Tensor BottleneckblockImpl::forward(torch::Tensor x){
    torch::Tensor identity = x;
    x = torch::relu(batchnorm1->forward(conv_1->forward(x)));
    x = torch::relu(batchnorm2->forward(conv_2->forward(x)));
    x = torch::relu(batchnorm3->forward(conv_3->forward(x)));
    x = torch::relu(torch::add(identity, x));
    return x;
}


//Various resnet architectures implementation, includes ResNet18, ResNet50, ResNet101, ResNet152
// https://github.com/pytorch/pytorch/issues/22298
// https://github.com/pytorch/pytorch/pull/23939

ResNetImpl::ResNetImpl(){}
ResNetImpl::ResNetImpl(std::string resnet_arch, std::string norm_type, std::vector<int> arch_layers,
        int input_channels, int groups, int width){
    conv_1 = register_module("conv_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 64, 7).stride(2).padding(2).
                bias(false)));
    maxpool_1 = register_module("maxpool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3, 3}).stride(2).padding(3)));
    c2 = register_module("c1", get_layer(resnet_arch, 0, 64, 64));
    c3 = register_module("c2", get_layer(resnet_arch, 1, 128, 128));
    c4 = register_module("c3", get_layer(resnet_arch, 2, 256, 256));
    c5 = register_module("c4", get_layer(resnet_arch, 3, 512, 512));

}
auto ResNetImpl::forward(torch::Tensor x){
    x = conv_1(x);
    torch::Tensor c1_x = maxpool_1(x);
    torch::Tensor c2_x = c2(x);
    torch::Tensor c3_x = c3(x);
    torch::Tensor c4_x = c4(x);
    torch::Tensor c5_x = c5(x);
    return std::make_tuple([c1_x, c2_x, c3_x, c4_x, c5_x]);
}

torch::nn::Sequential ResNetImpl::get_layer(std::string resnet_arch, int layer_num, int inplanes, int outplanes){
    std::unordered_map<std::string, std::vector<int>>arch_layer_nums;
    arch_layer_nums["resnet_18"] = std::vector<int>{2, 2, 2, 2};
    arch_layer_nums["resnet_34"] = std::vector<int>{3, 4, 6, 3};
    arch_layer_nums["resnet_50"] = std::vector<int>{3, 4, 6, 3};
    arch_layer_nums["resnet_101"] = std::vector<int>{3, 4, 23, 3};
    arch_layer_nums["resnet_151"] = std::vector<int>{3, 8, 36, 3};
    if(resnet_arch=="resnet_18" || resnet_arch=="resnet_34"){
      return   this->make_layer_using_basic_block(arch_layer_nums[resnet_arch][layer_num], inplanes, outplanes);
    }else{
       return this->make_layer_using_bottleneck_block(arch_layer_nums[resnet_arch][layer_num], inplanes, outplanes);
    }

}

torch::nn::Sequential ResNetImpl::make_layer_using_basic_block(int num, int inplanes, int outplanes){
    torch::nn::Sequential layers;
    for(int i =0; i < num; i++){
        layers->push_back(Basicblock(inplanes, outplanes));
    }   

}

torch::nn::Sequential ResNetImpl::make_layer_using_bottleneck_block(int num, int inplanes, int outplanes){
    torch::nn::Sequential layers;
    for(int i =0; i < num; i++){
        layers->push_back(Bottleneckblock(inplanes, outplanes));
    }   

}