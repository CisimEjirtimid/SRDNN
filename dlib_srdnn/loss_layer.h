#pragma once
#include <vector>
#include <dlib/matrix/matrix.h>
#include <dlib/dnn/loss.h>

#include "dnn_common.h"

namespace dnn
{
    namespace
    {
        // copied from dlib/input.h - input_rgb_image
        float   avg_red(122.782 / 256.0),
                avg_green(117.001 / 256.0),
                avg_blue(104.298 / 256.0);

        float average_color_from_index(int idx)
        {
            switch (idx)
            {
            case 0:
                return avg_red;
            case 1:
                return avg_green;
            case 2:
                return avg_blue;
            default:
                return 0;
            }
        }

        float clip_to_char(float input)
        {
            return std::min(255.0f, std::max(0.0f, input));
        }

        unsigned char channel_from_index(const dlib::rgb_pixel pix, int idx)
        {
            switch (idx)
            {
            case 0:
                return pix.red;
            case 1:
                return pix.green;
            case 2:
                return pix.blue;
            default:
                std::cout << "Invalid pixel index" << std::endl;
                return 0;
            }
        }

        void indexed_color_to_channel(dlib::rgb_pixel& pix, unsigned char color, int idx)
        {
            switch (idx)
            {
            case 0:
                pix.red = color;
                break;
            case 1:
                pix.green = color;
                break;
            case 2:
                pix.blue = color;
                break;
            default:
                std::cout << "Invalid pixel index" << std::endl;
                break;
            }
        }

        template<typename pixel_T>
        float truth_color(pixel_T pix, int idx)
        {
            std::cout << "Unknown image pixel format" << std::endl;
            return std::numeric_limits<float>::max();
        }

        template<>
        float truth_color<dlib::rgb_pixel> (dlib::rgb_pixel t, int idx)
        {
            return static_cast<float>(channel_from_index(t, idx));
        }

        template<>
        float truth_color<float>(float t, int idx)
        {
            DLIB_ASSERT(idx == 0);
            return t;
        }

        template<typename pixel_T>
        void output_color(pixel_T& pix, unsigned char color, int idx)
        {
            std::cout << "Unknown image pixel format" << std::endl;
        }

        template<>
        void output_color<dlib::rgb_pixel>(dlib::rgb_pixel& pix, unsigned char color, int idx)
        {
            indexed_color_to_channel(pix, color, idx);
        }

        template<>
        void output_color<float>(float& pix, unsigned char color, int idx)
        {
            DLIB_ASSERT(idx == 0);
            pix = color;
        }
    }

    class loss_pixel_
    {
    public:

        // In most cases training_label_type and output_label_type will be the same type.
        typedef dlib::matrix<pixel_type> training_label_type;
        typedef dlib::matrix<pixel_type>   output_label_type;

        loss_pixel_(
        )
        {
        }

        /*!
        ensures
        - EXAMPLE_LOSS_LAYER_ objects are default constructable.
        !*/

        loss_pixel_(
            const loss_pixel_& item
        )
        {
        }

        /*!
        ensures
        - EXAMPLE_LOSS_LAYER_ objects are copy constructable.
        !*/

        static size_t tensor_index(const dlib::tensor& t, long sample, long row, long column, long k);

        // Implementing to_label() is optional.
        template <
            typename SUB_TYPE,
            typename label_iterator
        >
        void to_label(
            const dlib::tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);

            const dlib::tensor& output_tensor = sub.get_output();
            //DLIB_CASSERT(output_tensor.k() == 3);
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

            auto out_data = output_tensor.host();

            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                output_label_type network_out(output_tensor.nr(), output_tensor.nc());

                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        for (long k = 0; k < output_tensor.k(); ++k)
                        {
                            auto idx = tensor_index(output_tensor, i, r, c, k);
                            output_color(network_out(r, c), clip_to_char(out_data[idx] * 255.0), k);
                        }
                    }
                }

                *iter++ = std::move(network_out);
            }
        }
        /*!
        requires
        - SUBNET implements the SUBNET interface defined at the top of
        layers_abstract.h.
        - input_tensor was given as input to the network sub and the outputs are
        now visible in layer<i>(sub).get_output(), for all valid i.
        - input_tensor.num_samples() > 0
        - input_tensor.num_samples()%sub.sample_expansion_factor() == 0.
        - iter == an iterator pointing to the beginning of a range of
        input_tensor.num_samples()/sub.sample_expansion_factor() elements.  Moreover,
        they must be output_label_type elements.
        ensures
        - Converts the output of the provided network to output_label_type objects and
        stores the results into the range indicated by iter.  In particular, for
        all valid i, it will be the case that:
        *(iter+i/sub.sample_expansion_factor()) is populated based on the output of
        sub and corresponds to the ith sample in input_tensor.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
        >
        double compute_loss_value_and_gradient(
            const dlib::tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub) const
        {
            const dlib::tensor& output_tensor = sub.get_output();
            dlib::tensor& grad = sub.get_gradient_input();

            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(input_tensor.num_samples() % sub.sample_expansion_factor() == 0);
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
            DLIB_CASSERT(output_tensor.k() == PIXEL_CHANNELS);
            DLIB_CASSERT(output_tensor.nr() == grad.nr() &&
                output_tensor.nc() == grad.nc());

            // The loss we output is the average loss over the mini-batch, and also over each element of the matrix output.
            const auto scale = 1.0 / (output_tensor.nr() * output_tensor.nc() * output_tensor.num_samples());

            double loss = 0;

            auto g = grad.host();
            auto out_data = output_tensor.host();

            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const_label_iterator truth_matrix_ptr = (truth + i);
                DLIB_CASSERT((*truth_matrix_ptr).nr() == output_tensor.nr() && (*truth_matrix_ptr).nc() == output_tensor.nc());

                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const pixel_type truth_pixel = (*truth_matrix_ptr)(r, c);

                        for (long k = 0; k < output_tensor.k(); k++)
                        {
                            auto idx = tensor_index(output_tensor, i, r, c, k);

                            auto ground_truth_color = static_cast<float>(truth_color(truth_pixel, k))/255.0;

                            auto error = ground_truth_color - out_data[idx];
                            loss += scale * error * error;
                            g[idx] = -2 * scale * error;
                        }
                    }
                }
            }
            return loss;
        }

        /*!
        requires
        - SUBNET implements the SUBNET interface defined at the top of
        layers_abstract.h.
        - input_tensor was given as input to the network sub and the outputs are
        now visible in layer<i>(sub).get_output(), for all valid i.
        - input_tensor.num_samples() > 0
        - input_tensor.num_samples()%sub.sample_expansion_factor() == 0.
        - for all valid i:
        - layer<i>(sub).get_gradient_input() has the same dimensions as
        layer<i>(sub).get_output().
        - truth == an iterator pointing to the beginning of a range of
        input_tensor.num_samples()/sub.sample_expansion_factor() elements.  Moreover,
        they must be training_label_type elements.
        - for all valid i:
        - *(truth+i/sub.sample_expansion_factor()) is the label of the ith sample in
        input_tensor.
        ensures
        - This function computes a loss function that describes how well the output
        of sub matches the expected labels given by truth.  Let's write the loss
        function as L(input_tensor, truth, sub).
        - Then compute_loss_value_and_gradient() computes the gradient of L() with
        respect to the outputs in sub.  Specifically, compute_loss_value_and_gradient()
        assigns the gradients into sub by performing the following tensor
        assignments, for all valid i:
        - layer<i>(sub).get_gradient_input() = the gradient of
        L(input_tensor,truth,sub) with respect to layer<i>(sub).get_output().
        - returns L(input_tensor,truth,sub)
        !*/
    };


    std::ostream& operator<<(std::ostream& out, const loss_pixel_& item);
    /*!
    print a string describing this layer.
    !*/

    void to_xml(const loss_pixel_& item, std::ostream& out);
    /*!
    This function is optional, but required if you want to print your networks with
    net_to_xml().  Therefore, to_xml() prints a layer as XML.
    !*/

    void serialize(const loss_pixel_& item, std::ostream& out);
    void deserialize(loss_pixel_& item, std::istream& in);
    /*!
    provides serialization support
    !*/

    // For each loss layer you define, always define an add_loss_layer template so that
    // layers can be easily composed.  Moreover, the convention is that the layer class
    // ends with an _ while the add_loss_layer template has the same name but without the
    // trailing _.
    template <typename SUBNET>
    using loss_pixel = dlib::add_loss_layer<loss_pixel_, SUBNET>;
}
