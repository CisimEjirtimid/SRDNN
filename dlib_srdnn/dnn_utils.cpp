#include "dnn_utils.h"
#include <dlib/image_transforms/interpolation.h>

namespace dnn
{
namespace utils
{
    namespace
    {
        // This function expects directory path as input and obtains a list of all the image files.
        std::vector<std::string> get_image_list(std::string& dir)
        {
            std::vector<std::string> images;

            for (auto img : dlib::directory(dir).get_files())
                images.push_back(img);

            return images;
        }
    }

    std::vector<dlib::matrix<dlib::rgb_pixel>> load_dataset(std::string& dir)
    {
        auto img_list = get_image_list(dir);

        std::vector<dlib::matrix<dlib::rgb_pixel>> images;

        int count = 0;
        for (auto& i : img_list)
        {
            dlib::matrix<dlib::rgb_pixel> img;
            dlib::load_image(img, i);
            images.push_back(img);

            if (++count % 100 == 0)
                std::cout << count << " images loaded." << std::endl;
        }

        return images;
    }

    std::vector<dlib::matrix<dlib::rgb_pixel>> resize_dataset(std::vector<dlib::matrix<dlib::rgb_pixel>>& dataset, double scale_factor)
    {
        std::vector<dlib::matrix<dlib::rgb_pixel>> upsampled;

        for (auto& img : dataset)
        {
            dlib::matrix<dlib::rgb_pixel> upsample(img.nr() * scale_factor, img.nc() * scale_factor);
            dlib::resize_image(img, upsample, dlib::interpolate_bilinear());
            upsampled.push_back(upsample);
        }

        return upsampled;
    }

    std::vector<dlib::matrix<dlib::rgb_pixel>> resize_dataset(std::vector<dlib::matrix<dlib::rgb_pixel>>& dataset, dlib::rectangle new_size)
    {
        std::vector<dlib::matrix<dlib::rgb_pixel>> upsampled;

	int count = 0;
        for (auto& img : dataset)
        {
            dlib::matrix<dlib::rgb_pixel> upsample(new_size.height(), new_size.width());
            dlib::resize_image(img, upsample, dlib::interpolate_bilinear());
            upsampled.push_back(upsample);
            
	    if (++count % 100 == 0)
                std::cout << count << " images resized." << std::endl;
        }

        return upsampled;
    }
}
}
