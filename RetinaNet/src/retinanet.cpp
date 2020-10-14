#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "retinanet.h"
#include "resnet.h"
#include "fpn.h"


RetinaHeadImpl::RetinaHeadImpl(){

}

RetinaHeadImpl::RetinaHeadImpl(int in_channels, int num_classes, int anchors_per_cell, 
            int num_convs, float prior=0.01, float drop_rate=0.01){
        // bbox subnet
        // clss subnet
        // bbox_pred, clss_pred
        torch::nn::Sequential bbox_subnet2;
        torch::nn::Sequential clss_subnet2;
        for(int i=0; i<num_convs; i++){
            bbox_subnet2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3)));
            bbox_subnet2->push_back(torch::nn::ReLU());

            clss_subnet2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3)));
            clss_subnet2->push_back(torch::nn::ReLU());
        } 

        bbox_subnet = register_module("bbox_subnet", bbox_subnet2);
        clss_subnet = register_module("bbox_subnet", clss_subnet2);

        bbox_pred = register_module("bbox_pred", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,
                anchors_per_cell*4,3)));
        clss_pred = register_module("clss_pred", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,
                anchors_per_cell * num_classes, 3)));
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> RetinaHeadImpl::forward(std::vector<torch::Tensor> feature_pyramids){
        std::vector<torch::Tensor> bbx_reg;
        std::vector<torch::Tensor> clss_logs;
        for(int i=0; i<feature_pyramids.size(); i++){
            bbx_reg.push_back(bbox_pred->forward(bbox_subnet->forward(feature_pyramids[i])));
            clss_logs.push_back(clss_pred->forward(clss_subnet->forward(feature_pyramids[i])));
        }

        return std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>(bbx_reg, clss_logs);
}

RetinaNetImpl::RetinaNetImpl(){

}
RetinaNetImpl::RetinaNetImpl(const std::string backbone_arch, const std::vector<int> pyramid_levels,
    int head_channels, int num_classes, int anchors_per_cell, int num_convs){
    // Right now only the backbone supported is ResNet style architecture, in the future we can add more backbones to this.
    if (backbone_arch.find("resnet")!=std::string::npos){
         backbone = std::make_shared<ResNet>(backbone_arch, "batchnorm", "", 3, 1, 64);
        // std::unique_ptr<ResNet> backbone = std::make_unique<ResNet>();
    }else{
        std::cout<<"Right now only ResNet architecture is supported for backbone"<<std::endl;
    }
    // FPN pointer to use it in the forward function

    std::vector<int> in_channels {512, 1024, 2048}; //we can get these directly from backbone, no need to hard code
    fpn_ptr = std::make_shared<FPN>(pyramid_levels, head_channels, in_channels);
    retinahead_ptr = std::make_shared<RetinaHead>(head_channels, anchors_per_cell,
         num_convs);
    
}

at::Tensor RetinaNetImpl::forward(torch::Tensor x){
    std::vector<torch::Tensor> backbone_features = backbone->forward(x);
    std::vector<torch::Tensor> feature_pyramids = fpn_ptr->forward(backbone_features);
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> preds = retinahead_ptr->forward(feature_pyramids);
    

}