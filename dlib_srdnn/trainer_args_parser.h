#pragma once
#include <string>
#include "dnn_setup.h"

namespace dnn
{
namespace args
{
    class parser
    {
        enum string_code {
            synchronization_file,
            learning_rate,
            min_learning_rate,
            mini_batch_size,
            iterations_without_progress_threshold
        };

        string_code code(std::string const& string) {
            if (string == "synchronization_file")                   return synchronization_file;
            if (string == "learning_rate")                          return learning_rate;
            if (string == "min_learning_rate")                      return min_learning_rate;
            if (string == "mini_batch_size")                        return mini_batch_size;
            if (string == "iterations_without_progress_threshold")  return iterations_without_progress_threshold;
        }

        void set_args(std::map<string, string>& args);

        dnn_trainer<sr_net>& trainer;
    public:
        explicit parser(dnn_trainer<sr_net>& trainer);
        bool parse(const string& args);
    };
}
}
