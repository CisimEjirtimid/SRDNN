#pragma once
#include <dlib/image_io.h>

namespace dnn
{
namespace utils
{
    std::vector < dlib::matrix<float>> load_dataset(std::string& dir);

    std::vector<dlib::matrix<float>> resize_dataset(std::vector<dlib::matrix<float>>& dataset, double scale_factor);

    std::vector<dlib::matrix<float>> resize_dataset(std::vector<dlib::matrix<float>>& dataset, dlib::rectangle new_size);

    void normImage(dlib::matrix<float>& src, float scale);

//    dlib::matrix<dlib::rgb_pixel> difference(const dlib::matrix<dlib::rgb_pixel>& first, const dlib::matrix<dlib::rgb_pixel>& second);
}
}
