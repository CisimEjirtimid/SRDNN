#pragma once
#include <dlib/image_io.h>

namespace dnn
{
namespace utils
{
    std::vector < dlib::matrix<dlib::rgb_pixel>> load_dataset(std::string& dir);

    std::vector<dlib::matrix<dlib::rgb_pixel>> resize_dataset(std::vector<dlib::matrix<dlib::rgb_pixel>>& dataset, double scale_factor);

    std::vector<dlib::matrix<dlib::rgb_pixel>> resize_dataset(std::vector<dlib::matrix<dlib::rgb_pixel>>& dataset, dlib::rectangle new_size);

    dlib::matrix<dlib::rgb_pixel> difference(const dlib::matrix<dlib::rgb_pixel>& first, const dlib::matrix<dlib::rgb_pixel>& second);
}
}
