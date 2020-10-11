#ifndef RETINANET_H
#define RETINANET_H

#include <torch/torch.h>
#include <vector>
#include <string>

class RetinaNetImpl{
    public:
        RetinaNetImpl();
        RetinaNetImpl(std::vector<int> pyramid_levels);
    private:
        // All the necessary operations needed for RetinaNet implmentation

};


#endif