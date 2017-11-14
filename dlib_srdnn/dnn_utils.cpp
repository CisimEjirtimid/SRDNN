#include "dnn_utils.h"
#include <dlib/image_transforms/interpolation.h>

using namespace dlib;

namespace dnn
{
namespace utils
{
    namespace
    {
        auto norm_factor = 1.0 / 255.0;

        // This function expects directory path as input and obtains a list of all the image files.
        std::vector<std::string> get_image_list(std::string& dir)
        {
            std::vector<std::string> images;

            for (auto img : dlib::directory(dir).get_files())
                images.push_back(img);

            return images;
        }

        void norm_pixel(rgb_pixel& pixel, float scale)
        {
            pixel.red *= scale;
            pixel.green *= scale;
            pixel.blue *= scale;
        }

        void norm_pixel(float& pixel, float scale)
        {
            pixel *= scale;
        }

        rgb_pixel pixel_diff(const rgb_pixel& first, const rgb_pixel& second)
        {
            rgb_pixel diff;

            diff.red = first.red - second.red;
            diff.green = first.green - second.green;
            diff.blue = first.blue - second.blue;

            return diff;
        }

        float pixel_diff(float first, float second)
        {
            return first - second;
        }
    }

    std::vector<dlib::matrix<pixel_type>> load_dataset(std::string& dir)
    {
        auto img_list = get_image_list(dir);

        std::vector<dlib::matrix<pixel_type>> images;

        auto count = 0;
        for (auto& i : img_list)
        {
            dlib::matrix<pixel_type> img;
            dlib::load_image(img, i);

            if (std::is_same<pixel_type, rgb_pixel>())
            {
                //norm_image(img, norm_factor); rgb values are chars, not floats, so with norming the information is lost
                images.push_back(img);
            }
            else if (std::is_same<pixel_type, float>())
            {
                dlib::matrix<pixel_type> img_converted;
                dlib::assign_image(img_converted, img);

                norm_image(img_converted, 1.0 / 255.0);

                images.push_back(img_converted);
            }
            else
            {
                std::cout << "Unknown image pixel format" << std::endl;
            }

            if (++count % 100 == 0)
                std::cout << count << " images loaded." << std::endl;
        }

        return images;
    }

    std::vector<dlib::matrix<pixel_type>> resize_dataset(std::vector<dlib::matrix<pixel_type>>& dataset, double scale_factor)
    {
        std::vector<dlib::matrix<pixel_type>> upsampled;

        for (auto& img : dataset)
        {
            dlib::matrix<pixel_type> upsample(img.nr() * scale_factor, img.nc() * scale_factor);
            dlib::resize_image(img, upsample, dlib::interpolate_bilinear());
            upsampled.push_back(upsample);
        }

        return upsampled;
    }

    std::vector<dlib::matrix<pixel_type>> resize_dataset(std::vector<dlib::matrix<pixel_type>>& dataset, dlib::rectangle new_size)
    {
        std::vector<dlib::matrix<pixel_type>> upsampled;

        auto count = 0;

        for (auto& img : dataset)
        {
            dlib::matrix<pixel_type> upsample(new_size.height(), new_size.width());
            dlib::resize_image(img, upsample, dlib::interpolate_bilinear());
            upsampled.push_back(upsample);
            
            if (++count % 100 == 0)
                std::cout << count << " images resized." << std::endl;
        }

        return upsampled;
    }

    void norm_image(dlib::matrix<pixel_type>& img_gray, float scale)
    {
        for (auto i = 0; i < img_gray.nr(); i++)
            for (auto j = 0; j < img_gray.nc(); j++)
                norm_pixel(img_gray(i, j), scale);
    }

    void norm_dataset(std::vector<dlib::matrix<pixel_type>>& src_dataset, float scale)
    {
        for (auto i = 0; i < src_dataset.size(); i++)
            norm_image(src_dataset[i], scale);
    }

    dlib::matrix<pixel_type> difference(const dlib::matrix<pixel_type>& first, const dlib::matrix<pixel_type>& second)
    {
        DLIB_CASSERT(first.nr() == second.nr());
        DLIB_CASSERT(first.nc() == second.nc());

        auto res = dlib::matrix<pixel_type>(first.nr(), first.nc());

        for (auto i = 0; i < first.nr(); i++)
        {
            for(auto j = 0; j < first.nc(); j++)
            {
                auto first_pix = first(i, j);
                auto second_pix = second(i, j);

                res(i, j) = pixel_diff(first_pix, second_pix);
            }
        }

        return res;
    }

    float square_difference(const dlib::matrix<pixel_type>& first, const dlib::matrix<pixel_type>& second)
    {
        dlib::matrix<float> diff_gr;
        auto diff = difference(first, second);
        assign_image(diff_gr, diff);
        auto square_diff = pointwise_multiply(diff_gr, diff_gr);
        return sqrt(float(sum(square_diff)));
    }
}
}
