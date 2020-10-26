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
    // grid_sizes -->  [(512/8, 512/8),(512/16, 512/16),(512/32, 512/32),(512/64, 512/64),(512/128, 512/128)]
    // anchors_per_cell --> [anchorssize (16)--> 9 anchorboxes, 32 --> 9 anchorboxes, .....]
    
    float anchor_offset = 0.5;
    
    for(int sz=0; sz<grid_sizes.size(); sz++){
        int grid_height = grid_sizes[sz].first;
        int grid_width = grid_sizes[sz].second;
        torch::Tensor shifts_x = torch::arange(anchor_offset * feature_strides[sz],
            grid_width * feature_strides[sz], feature_strides[sz], 
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        torch::Tensor shifts_y = torch::arange(anchor_offset * feature_strides[sz],
            grid_height * feature_strides[sz], feature_strides[sz], 
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        
        std::vector<torch::Tensor> shifts_v{shifts_y, shifts_x};
        torch::TensorList tensor_lst = shifts_v;
        shifts_v = torch::meshgrid(tensor_lst);
        std::vector<torch::Tensor> shifts_x_y{shifts_v[1], shifts_v[0], shifts_v[1], shifts_v[0]};
        tensor_lst = shifts_x_y;
        torch::Tensor shifts = torch::stack(tensor_lst).view({-1 ,1, 4});
        torch::Tensor base_anchor = torch::from_blob(anchors_per_cell[anchor_sizes[sz]].data(), 
                {int(anchors_per_cell.size())}).view({-1 ,1, 4});
        final_anchors.push_back(torch::add(shifts, base_anchor));
        
    }

}

int Anchors::get_anchors_per_cell(){
    return anchor_scales.size() * aspect_ratios.size();
}

std::vector<torch::Tensor> Anchors::get_all_anchors(){
    return final_anchors;
}