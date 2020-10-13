#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "retinanet.h"
#include "resnet.h"
#include "fpn.h"

RetinaNetImpl::RetinaNetImpl(){

}

RetinaNetImpl::RetinaNetImpl(const std::string backbone_arch, const std::vector<int> pyramid_levels){
    // Right now only the backbone supported is ResNet style architecture, in the future we can add more backbones to this.
    if (backbone_arch.find("resnet")!=std::string::npos){
         backbone = std::make_shared<ResNet>(backbone_arch, "batchnorm", "", 3, 1, 64);
        // std::unique_ptr<ResNet> backbone = std::make_unique<ResNet>();
    }else{
        std::cout<<"Right now only ResNet architecture is supported for backbone"<<std::endl;
    }
    // FPN pointer to use it in the forward function

    std::vector<int> in_channels {512, 1024, 2048}; //we can get these directly from backbone, no need to hard code
    std::shared_ptr<FPN> fpn_ptr = std::make_shared<FPN>(pyramid_levels, 128, in_channels);

    

}

at::Tensor RetinaNetImpl::forward(torch::Tensor x){

}