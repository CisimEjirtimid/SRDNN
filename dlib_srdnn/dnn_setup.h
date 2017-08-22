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
    #define SR_SCALE 2

    // output = 3 filters - one for each channel in pixel
    template<
        typename SUBNET
    >
    using output_block = relu<con<1, 1, 1, 1, 1, SUBNET>>;

    // input = bilinear upsample of the input image
    template<
        int scale
    >
    using input_block = upsample<scale, dlib::input<matrix<float>>>;

    namespace batch_normalised /* unused */
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
        using bn_block = relu<BN<con<num_filters, conv_size, conv_size, 1, 1, SUBNET>>>;

        // self-explanatory
        template <
            template<typename> class BN,
            typename SUBNET
        >
        using bn_con_block =
            bn_block<64, 3, BN,
            bn_block<64, 5, BN,
            bn_block<64, 7, BN,
            bn_block<64, 9, BN,
            bn_block<64, 11, BN,
            bn_block<64, 13, BN,
            bn_block<64, 15, BN,
            bn_block<64, 17, BN,
            bn_block<64, 19, BN,
            bn_block<64, 21, BN,
            bn_block<64, 23, BN,
            bn_block<64, 25, BN,
            bn_block<64, 27, BN,
            SUBNET>>>>>>>>>>>>>;

        // residual creates a network structure like this:
        /*
             input from SUBNET
                 /     \
                /       \
              block    skip
                \       /
                 \     /
               add tensors (using add_prev1 which adds the output of tag1)
                    |
                 output
        */
        template <
            template<typename> class BN,
            template<template<typename> class, typename> class block,
            typename SUBNET
        >
        using bn_residual = add_prev1<block<BN, tag1<SUBNET>>>;

        template<
            int scale,
            template<typename> class BN
        >
        using bn_net_def =
            loss_pixel<
            BN<
            output_block<
            bn_residual<BN,
            bn_con_block,
            input_block<scale>>>>>;

        using sr_net = bn_net_def<2, bn_con>;

        using sr_eval_net = bn_net_def<2, affine>;
    }

    template<
        int num_filters,
        int conv_size,
        typename SUBNET
    >
    using block = relu<con<num_filters, conv_size, conv_size, 1, 1, SUBNET>>;

    // self-explanatory
    template <
        typename SUBNET
    >
    using con_block =
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        SUBNET>>>>>>>>>>>>>;

    template <
        typename SUBNET
    >
    using simple_con_block =
        output_block<
        block<64, 3,
        block<64, 3,
        block<64, 3,
        block<64, 3,
        SUBNET>>>>>;

    // residual creates a network structure like this:
    /*
        input from SUBNET
            /     \
           /       \
         block    skip
           \       /
            \     /
         add tensors (using add_prev1 which adds the output of tag1)
               |
            output
    */
    template <
        template<typename> class block,
        typename SUBNET
    >
    using residual = add_prev1<block<tag1<SUBNET>>>;

    template<
        int scale
    >
    using net_def =
        loss_pixel<output_block<residual<con_block, input_block<scale>>>>;

    template<
        int scale
    >
    using simple_net_def =
        loss_pixel<
        output_block<
        residual<
        simple_con_block,
        input_block<scale>>>>;

    using sr_net = net_def<SR_SCALE>;

    using simple_net = simple_net_def<SR_SCALE>;
}
