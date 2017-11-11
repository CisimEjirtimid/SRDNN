#pragma once
#include <dlib/image_io.h>

#include "dnn_common.h"

namespace dnn
{
namespace utils
{
    std::vector <dlib::matrix<pixel_type>> load_dataset(std::string& dir);

    std::vector<dlib::matrix<pixel_type>> resize_dataset(std::vector<dlib::matrix<pixel_type>>& dataset, double scale_factor);
    std::vector<dlib::matrix<pixel_type>> resize_dataset(std::vector<dlib::matrix<pixel_type>>& dataset, dlib::rectangle new_size);

    void norm_image(dlib::matrix<pixel_type>& src, float scale);
    void norm_dataset(std::vector<dlib::matrix<pixel_type>>& src_dataset, float scale);

    dlib::matrix<pixel_type> difference(const dlib::matrix<pixel_type>& first, const dlib::matrix<pixel_type>& second);
    float square_difference(const dlib::matrix<pixel_type>& first, const dlib::matrix<pixel_type>& second);
}
}
