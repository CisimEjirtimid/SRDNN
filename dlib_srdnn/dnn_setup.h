#pragma once
#include <dlib/dnn.h>
#include <dlib/dnn/layers.h>
#include <iostream>
#include <dlib/data_io.h>
#include "loss_layer.h"

using namespace std;
using namespace dlib;

namespace dnn
{
    // rbc = Convolution - Batch Normalization - ReLU -- block -- for network training
    template<
        int num_filters,
        int conv_size,
        typename SUBNET
    >
    using rbc_block = relu<bn_con<con<num_filters, conv_size, conv_size, 1, 1, SUBNET>>>;

    // rac = Convolution - Affine - ReLU -- block -- for network evalation
    template<
        int num_filters,
        int conv_size,
        typename SUBNET
    >
    using rac_block = relu<affine<con<num_filters, conv_size, conv_size, 1, 1, SUBNET>>>;

    template<
        int num_filters,
        int conv_size,
        template<typename> class BN,
        typename SUBNET
    >
    using block = relu<BN<con<num_filters, conv_size, conv_size, 1, 1, SUBNET>>>;

    // output = 3 filters - one for each channel in pixel
    template<
        typename SUBNET
    >
    using output_block = relu<con<3, 1, 1, 1, 1, SUBNET>>;

    // input = bilinear upsample of the input image
    template<
        int scale
    >
    using input_block = upsample<scale, input_rgb_image>;

    // self-explanatory
    template <
        template<typename> class BN,
        typename SUBNET
    >
    using con_block = 
        block<64,   3,  BN,
        block<64,   5,  BN,
        block<64,   7,  BN,
        block<64,   9,  BN,
        block<64,   11, BN,
        block<64,   13, BN,
        block<64,   15, BN,
        block<64,   17, BN,
        block<64,   19, BN,
        block<64,   21, BN,
        block<64,   23, BN,
        block<64,   25, BN,
        block<64,   27, BN,
        SUBNET>>>>>>>>>>>>>;

    // residual creates a network structure like this:
    /*
         input from SUBNET
             /     \
            /       \
       con_block    skip
            \       /
             \     /
           add tensors (using add_prev1 which adds the output of tag1)
                |
             output
    */
    template <
        template<template<typename> class, typename> class block,
        template<typename> class BN,
        typename SUBNET
    >
    using residual = add_prev1<block<BN, tag1<SUBNET>>>;

    template<
        template<typename> class BN
    >
    using net_def = 
        loss_pixel<
        output_block<
        residual<BN,
        con_block,
        input_block<2>>>>;

    using sr_net = net_def<bn_con>;

    using sr_eval_net = net_def<affine>;
}