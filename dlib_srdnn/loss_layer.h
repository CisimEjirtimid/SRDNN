#pragma once
#include <vector>
#include <dlib/pixel.h>
#include <dlib/matrix/matrix.h>
#include <dlib/dnn/loss.h>

namespace dnn
{
    namespace
    {
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
            }
            return 0;
        }
    }

    class loss_avg_
    {
    public:

        // In most cases training_label_type and output_label_type will be the same type.
        typedef dlib::matrix<dlib::rgb_pixel> training_label_type;
        typedef dlib::matrix<dlib::rgb_pixel>   output_label_type;

        loss_avg_(
        )
        {
        }

        /*!
        ensures
        - EXAMPLE_LOSS_LAYER_ objects are default constructable.
        !*/

        loss_avg_(
            const loss_avg_& item
        )
        {
        }

        /*!
        ensures
        - EXAMPLE_LOSS_LAYER_ objects are copy constructable.
        !*/

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

        static size_t tensor_index(const dlib::tensor& t, long sample, long row, long column, long k);

        //Y = 0.299R + 0.587G + 0.114B
        //U = -0.147R - 0.289G + 0.436B
        //V = 0.615R - 0.515G - 0.100B
        //static rgb_to_yuv

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
            //DLIB_CASSERT(output_tensor.k() == 3); // rgb_pixel
            DLIB_CASSERT(output_tensor.nr() == grad.nr() &&
                output_tensor.nc() == grad.nc());

            for (long idx = 0; idx < output_tensor.num_samples(); ++idx)
            {
                const_label_iterator truth_matrix_ptr = (truth + idx);
                DLIB_CASSERT((*truth_matrix_ptr).nr() == output_tensor.nr() &&
                    (*truth_matrix_ptr).nc() == output_tensor.nc());
            }

            // The loss we output is the average loss over the mini-batch, and also over each element of the matrix output.
            const double scale = 1.0 / (output_tensor.num_samples() * output_tensor.nr() * output_tensor.nc() * output_tensor.k());
            double loss = 0;
            float* const g = grad.host();

            auto iter = output_tensor.begin();

            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const dlib::rgb_pixel y = truth->operator()(r, c);

                        for (long k = 0; k < output_tensor.k(); k++)
                        {
                            const size_t idx = tensor_index(output_tensor, i, r, c, k);
                            auto output = static_cast<unsigned char>(*(iter + idx) * 255);

                            auto truth_color = channel_from_index(y, k);

                            auto diff = truth_color - output;
                            loss += scale * (diff);
                            g[idx] = -diff;
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


    std::ostream& operator<<(std::ostream& out, const loss_avg_& item);
    /*!
    print a string describing this layer.
    !*/

    void to_xml(const loss_avg_& item, std::ostream& out);
    /*!
    This function is optional, but required if you want to print your networks with
    net_to_xml().  Therefore, to_xml() prints a layer as XML.
    !*/

    void serialize(const loss_avg_& item, std::ostream& out);
    void deserialize(loss_avg_& item, std::istream& in);
    /*!
    provides serialization support
    !*/

    // For each loss layer you define, always define an add_loss_layer template so that
    // layers can be easily composed.  Moreover, the convention is that the layer class
    // ends with an _ while the add_loss_layer template has the same name but without the
    // trailing _.
    template <typename SUBNET>
    using loss_avg = dlib::add_loss_layer<loss_avg_, SUBNET>;
}
