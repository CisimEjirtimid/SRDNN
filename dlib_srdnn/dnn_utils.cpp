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

    std::vector<dlib::matrix<float>> load_dataset(std::string& dir)
    {
        auto img_list = get_image_list(dir);

        std::vector<dlib::matrix<float>> images;

        int count = 0;
        for (auto& i : img_list)
        {
            dlib::matrix<dlib::rgb_pixel> img;
            dlib::load_image(img, i);
            dlib::matrix<float> img_gray;
            dlib::assign_image(img_gray, img);

            normImage(img_gray, 1.0/255.0);

            //img_gray *= 1/255.0;

            images.push_back(img_gray);

            if (++count % 100 == 0)
                std::cout << count << " images loaded." << std::endl;
        }

        return images;
    }

    std::vector<dlib::matrix<float>> resize_dataset(std::vector<dlib::matrix<float>>& dataset, double scale_factor)
    {
        std::vector<dlib::matrix<float>> upsampled;

        for (auto& img : dataset)
        {
            dlib::matrix<float> upsample(img.nr() * scale_factor, img.nc() * scale_factor);
            dlib::resize_image(img, upsample, dlib::interpolate_bilinear());
            upsampled.push_back(upsample);
        }

        return upsampled;
    }

    std::vector<dlib::matrix<float>> resize_dataset(std::vector<dlib::matrix<float>>& dataset, dlib::rectangle new_size)
    {
        std::vector<dlib::matrix<float>> upsampled;

        int count = 0;

        for (auto& img : dataset)
        {
            dlib::matrix<float> upsample(new_size.height(), new_size.width());
            dlib::resize_image(img, upsample, dlib::interpolate_bilinear());
            upsampled.push_back(upsample);
            
            if (++count % 100 == 0)
                std::cout << count << " images resized." << std::endl;
        }

        return upsampled;
    }
    

    void normImage(dlib::matrix<float>& img_gray, float scale)
    {
            for (auto i = 0; i < img_gray.nr(); i++)
                for (auto j = 0; j < img_gray.nc(); j++)
                    img_gray(i, j) *= scale;
    }

    /*
    dlib::matrix<dlib::rgb_pixel> difference(const dlib::matrix<dlib::rgb_pixel>& first, const dlib::matrix<dlib::rgb_pixel>& second)
    {
        DLIB_CASSERT(first.nr() == second.nr());
        DLIB_CASSERT(first.nc() == second.nc());

        auto res = dlib::matrix<dlib::rgb_pixel>(first.nr(), first.nc());

        for (auto i = 0; i < first.nr(); i++)
        {
            for(auto j = 0; j < first.nc(); j++)
            {
                auto first_pix = first(i, j);
                auto second_pix = second(i, j);

                res(i, j).red = first_pix.red - second_pix.red;
                res(i, j).green = first_pix.green - second_pix.green;
                res(i, j).blue = first_pix.blue - second_pix.blue;
            }
        }

        return res;
    }
    */
}
}
