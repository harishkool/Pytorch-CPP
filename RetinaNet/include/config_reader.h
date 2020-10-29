#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <iostream>
#include <vector>
#include <string>
#include "json.h"

namespace config{

    class ConfigReader{

        private:
            struct Backbone final{
                std::string backbone_arch;
                std::vector<int> levels;
            };

            struct Head final{
                std::string pyramid;
                int num_convs;
                int head_channels;
                std::vector<int> pyramid_levels;
            };

            struct AnchorConfig final{
                std::vector<int> sizes;
                std::vector<float> aspect_ratios;
                std::vector<int> scales;
            };

            struct TrainingParameters final{
                std::string optimizer;
                int epochs;
                std::string save_dir;
                float initial_lr;
                float max_lr;
                std::string lr_decay;
            };

        public:
            ConfigReader(const std::string config_path);
            Backbone backbone;
            Head head;
            AnchorConfig anchor_config;
            TrainingParameters training_param;
            virtual ~ConfigReader();
    };



}

#endif


