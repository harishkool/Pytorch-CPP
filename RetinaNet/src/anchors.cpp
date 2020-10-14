#include "anchors.h"
#include <iostream>
#include <vector>

Anchors::Anchors(std::vector<int> aspect_ratios, std::vector<int> anchor_scales,
            std::vector<int> feature_strides, std::pair<int, int> input_size, std::vector<int> anchor_sizes){
    this->anchor_scales = anchor_scales;
    this->anchor_sizes = anchor_sizes;
    this->aspect_ratios = aspect_ratios;

}

int Anchors::get_anchors_per_cell(){
    return anchor_scales.size() * aspect_ratios.size();
}