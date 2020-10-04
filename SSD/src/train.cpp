#include <torch/torch.h>
#include <iostream>


int main(){

    torch::Tensor t1 = torch::ones({2,3});
    std::cout<<t1<<std::endl;
    return 0;

}
