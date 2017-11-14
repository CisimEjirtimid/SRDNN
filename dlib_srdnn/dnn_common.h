#pragma once
#include <dlib/pixel.h>

namespace dnn
{
    #define SR_SCALE 2

#ifdef _RGB
    using pixel_type = dlib::rgb_pixel;
    #define PIXEL_CHANNELS 3
#elif _GREY
    using pixel_type = float;
    #define PIXEL_CHANNELS 1
#endif
}
