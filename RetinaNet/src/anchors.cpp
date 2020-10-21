#include "anchors.h"
#include <iostream>
#include <vector>
#include <cmath>

Anchors::Anchors(std::vector<int> aspect_ratios, std::vector<int> anchor_scales,
            std::vector<int> feature_strides, std::pair<int, int> input_size, std::vector<int> anchor_sizes){
    this->anchor_scales = anchor_scales;
    this->anchor_sizes = anchor_sizes;
    this->aspect_ratios = aspect_ratios;
    this->feature_strides = feature_strides;

    for (int sz =0; sz < anchor_sizes.size(); sz++){
        std::vector<std::vector<int>> anchor_scale;
        for(int scl=0; scl < anchor_scales.size(); scl++){
            float area = pow((anchor_sizes[sz] * anchor_scales[scl]), 2.0); 
            for(int asp=0; asp < aspect_ratios.size(); asp++){
                std::vector<int> anchor;
                float w = sqrt(area / aspect_ratios[asp]);
                float h = aspect_ratios[asp] * w;
                float x0 = -w / 2.0;
                float y0 = -h / 2.0;
                float x1 = w / 2.0;
                float y1 = h / 2.0;
                anchor.push_back(x0);
                anchor.push_back(y0);
                anchor.push_back(x1);
                anchor.push_back(y1);
                anchor_scale.push_back(anchor);
            }

        }

        anchors_per_cell.insert(std::pair<int, std::vector<std::vector<int>>>(anchor_sizes[sz], anchor_scale));
    }

    std::vector<std::pair<int, int>> grid_sizes;
    auto func = [input_size](int s){
        return std::pair<int, int>(int(input_size.first/s), int(input_size.second/s));
    };

    for(auto s:feature_strides){
        grid_sizes.push_back(std::pair<int, int>(int(input_size.first/s), int(input_size.second/s)));
    }
    // 512, 512 --> input size
    // 2 ** level for level in pyramid_level --> [3, 4, 5, 6, 7]
    // strides --> [8, 16, 32, 64, 128]
    // anchor sizes --> [16, 32, 64, 128, 256]
    

}

int Anchors::get_anchors_per_cell(){
    return anchor_scales.size() * aspect_ratios.size();
}