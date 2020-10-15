#ifndef ANCHORS_H
#define ANCHORS_H

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>

class Anchors{
    public:
        Anchors(std::vector<int> aspect_ratios, std::vector<int> anchor_scales,
            std::vector<int> feature_strides, std::pair<int, int> input_size, std::vector<int> anchor_sizes);

        int get_anchors_per_cell();
    private:
    int anchors_per_cell;
    std::vector<int> anchor_sizes;
    std::vector<int> anchor_scales;
    std::vector<int> aspect_ratios;
    // std::vector<std::pair<int, std::vector<int>> > anchors; 
    std::unordered_map<int, std::vector<std::vector<int>>> anchors;
    
};



#endif