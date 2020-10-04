#ifndef VGGSSD_H
#define VGGSSD_H

#include<torch/torch.h>


class VggSSDImpl :public torch::nn::Module{
    public:
        VggSSDImpl();
        VggSSDImpl(int num_classes);
        torch::Tensor forward(torch::Tensor x);
    private:
        int num_classes;
        torch::nn::Conv2d layer1{nullptr};
            
};

TORCH_MODULE(VggSSD);


#endif