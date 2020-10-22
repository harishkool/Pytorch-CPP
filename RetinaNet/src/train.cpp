#include<torch/torch.h>
#include <vector>
#include <memory>
#include <cmath>

int main(){
    // all data loader code here
    // defining the model
    // defining the optimizer
    // training loop
    // evaluation loop
    // get the model config from a json or yaml file
    std::vector<int> pyramid_levels {3, 4, 5, 6, 7};
    int input_height = 512;
    int input_width = 512;
    std::vector<int> strides;

    for(int i=0;i < pyramid_levels.size();i++){
        strides.push_back(std::pow(2, pyramid_levels[i]));
    }
    

    return 0;
}


