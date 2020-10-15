#include "anchors.h"
#include <iostream>
#include <vector>
#include <cmath>

Anchors::Anchors(std::vector<int> aspect_ratios, std::vector<int> anchor_scales,
            std::vector<int> feature_strides, std::pair<int, int> input_size, std::vector<int> anchor_sizes){
    this->anchor_scales = anchor_scales;
    this->anchor_sizes = anchor_sizes;
    this->aspect_ratios = aspect_ratios;

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

        anchors.insert(std::pair<int, std::vector<std::vector<int>>>(anchor_sizes[sz], anchor_scale));
    }
}

int Anchors::get_anchors_per_cell(){
    return anchor_scales.size() * aspect_ratios.size();
}