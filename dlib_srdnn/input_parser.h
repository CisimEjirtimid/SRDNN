#pragma once

#include <argagg/argagg.hpp>

namespace dnn
{
namespace input
{
    argagg::parser parser {{
        { "help", { "-h", "--help" },
        "Shows the help message", 0 },

        { "train", { "-t", "--train" },
        "Begin neural network training", 0 },

        { "trainer-arguments", { "-a", "--trainer-arguments" },
        "Arguments for the DNN trainer, passed in <parameter_1>=<value_1>:<parameter_2>:<value_2> format", 1 },

        { "valid", { "-v", "--valid" },
        "Validation testing trained network, with provided validation dataset", 0 },

        { "eval", { "-e", "--eval" },
        "Evaluate trained network", 0 },

        { "input", { "-i", "--input" },
        "Input - dataset for training, or image for evaluation process", 1 },

        { "net-input", { "-n", "--net-input" },
        "Saved network params input for validation & evaluation process", 1 },

        { "output", { "-o", "--output" },
        "Output path - for saving trained network, or evaluated image", 1 },

        { "xml", { "-x", "--xml" },
        "Output path for saving trained network as XML", 1 },

        { "show", { "-s", "--show" },
        "Show the results of the evaluation", 0 },
    }};
}
}