#include "config_reader.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include "json.h"
#include <fstream>

namespace config{
    
    ConfigReader::ConfigReader(std::string config_filepath){
        
        Json::Value network_config;
        std::ifstream config_doc(config_filepath, std::ifstream::binary);
        config_doc >> network_config;
        config_doc.close();

        // Getting the backbone details
        Json::Value backbone_det = network_config['backbone'];
        backbone.backbone_arch = backbone_det.get('arch', 'resnet_18').asString();
        for(auto lvl: backbone_det['levels']){
            backbone.levels.push_back(lvl.asInt());
        }

        // Getting the head details
        Json::Value head_det = network_config['head'];
        head.head_channels = head_det.get('head_channels', 128).asInt();
        head.num_convs = head_det.get('num_convs', 5).asInt();
        head.pyramid = head_det.get('pyramid', 'fpn').asString();
        for(auto lvl : head_det['pyramid_levels']){
            head.pyramid_levels.push_back(lvl.asInt());
        }

        // Getting the anchor details
        Json::Value anchor_det = network_config['anchor'];
        for(auto asp : anchor_det['aspect_ratios']){
            anchor_config.aspect_ratios.push_back(asp.asFloat());
        }  

        for(auto sz : anchor_det['sizes']){
            anchor_config.sizes.push_back(sz.asInt());
        }

        for(auto scl : anchor_det['scales']){
            anchor_config.scales.push_back(scl.asInt());
        }

        // Getting the training parameters
        training_param.epochs = network_config['num_epochs'].asInt();
        training_param.initial_lr = network_config['initial_lr'].asFloat();
        training_param.max_lr = network_config['max_lr'].asFloat();
        training_param.lr_decay = network_config['decay'].asString();
        training_param.optimizer = network_config['optimizer'].asString();
        training_param.save_dir = network_config['save'].asString();
    }

    ConfigReader::~ConfigReader(){

    }

}