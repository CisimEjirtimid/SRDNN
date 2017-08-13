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
    //using sr_net = loss_avg<
    //    relu<con<3, 1, 1, 1, 1,
    //    relu<con<16, 3, 3, 1, 1,
    //    relu<con<256, 5, 5, 1, 1,
    //    relu<con<256, 7, 7, 1, 1,
    //    input_rgb_image>>>>>>>>>;

    //using sr_net = loss_avg<
    //    relu<con<3,  1, 1, 1, 1,
    //    relu<con<60, 3, 3, 1, 1,
    //    relu<con<30, 5, 5, 1, 1,
    //    input<matrix<rgb_pixel>>>>>>>>>;

    //using sr_net = loss_avg<
    //    relu<con<3, 1, 1, 1, 1,
    //    relu<con<64, 3, 3, 1, 1,
    //    relu<con<64, 5, 5, 1, 1,
    //    relu<con<64, 7, 7, 1, 1,
    //    relu<con<64, 9, 9, 1, 1,
    //    relu<con<64, 11, 11, 1, 1,
    //    relu<con<64, 13, 13, 1, 1,
    //    relu<con<64, 15, 15, 1, 1,
    //    relu<con<64, 17, 17, 1, 1,
    //    relu<con<64, 19, 19, 1, 1,
    //    relu<con<64, 21, 21, 1, 1,
    //    relu<con<64, 23, 23, 1, 1,
    //    relu<con<64, 25, 25, 1, 1,
    //    relu<con<64, 27, 27, 1, 1,
    //    input_rgb_image>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

    using input_layer = upsample<2, input_rgb_image>;

    template <
        int N,
        typename SUBNET
    >
    using con_block = 
        relu<bn_con<con<3,  1,  1,  1,  1,
        relu<bn_con<con<N,  3,  3,  1,  1,
        relu<bn_con<con<N,  5,  5,  1,  1,
        relu<bn_con<con<N,  7,  7,  1,  1,
        relu<bn_con<con<N,  9,  9,  1,  1,
        relu<bn_con<con<N,  11, 11, 1,  1,
        relu<bn_con<con<N,  13, 13, 1,  1,
        relu<bn_con<con<N,  15, 15, 1,  1,
        relu<bn_con<con<N,  17, 17, 1,  1,
        relu<bn_con<con<N,  19, 19, 1,  1,
        relu<bn_con<con<N,  21, 21, 1,  1,
        relu<bn_con<con<N,  23, 23, 1,  1,
        relu<bn_con<con<N,  25, 25, 1,  1,
        relu<bn_con<con<N,  27, 27, 1,  1,
        SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

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
        int N,
        typename SUBNET
    >
    using residual = add_prev1<con_block<N, tag1<SUBNET>>>;

    using sr_net =
        loss_avg<
        residual<64,
        input_layer>>;
}