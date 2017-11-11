#include <dlib/dnn.h>

#include "input_parser.h"
#include "dnn_utils.h"
#include "trainer_args_parser.h"
#include "compute_vifp.h"

using namespace dlib;
using namespace std;
using namespace dnn;

#include <dlib/gui_core.h>
#include <dlib/gui_widgets.h>

/*
    main.cpp file for dlib_srdnn project

    Usage: try dlib_srdnn --help for it

    Supported super-resolution scales: theoretically infinite
    Supported pixel types: rgb_pixel and float

    Setup:
        - Current super-resolution scale: 2
        - Current pixel type: rgb_pixel
            - Number of channels: 3

        - This parameters can be changed in dnn_common.h

    For usage with some other network:
        - Currently the simple training and evaluation framework wraps around the SRDNN,
          but could be used with any network architecture, just by changing the network in the dnn_setup.h file
*/

namespace
{
    void print_help()
    {
        cerr << "Simple SRDNN training/evaluation program" << endl;
        cerr << dnn::input::parser;
    }

    void training(argagg::parser_results args)
    {
        if (!bool(args["input"]))
        {
            cerr << "Invalid usage! Input is missing!" << endl;
            cerr << dnn::input::parser;
        }

        string str(args["input"].as<string>());

        auto images = utils::load_dataset(str);

        auto compatible_rect = rectangle(images[0].nc() + images[0].nc() % SR_SCALE, images[0].nr() + images[0].nr() % SR_SCALE);
        images = utils::resize_dataset(images, compatible_rect);
        auto downsampled = utils::resize_dataset(images, 1.0f / SR_SCALE);

        //save_jpeg(downsampled[0], "downsampled.jpg");
        //save_jpeg(images[0], "original.jpg");

        sr_net dnnet;
        dnn_trainer<sr_net> trainer(dnnet);
        args::parser args_parser(trainer);
        if (args["trainer_args"])
            args_parser.parse(args["trainer_args"].as<string>());

        trainer.be_verbose();
        trainer.train(downsampled, images);

        dnnet.clean();

        if (args["output"])
            serialize(args["output"].as<string>()) << dnnet;

        if (args["xml"])
            net_to_xml(dnnet, args["xml"].as<string>());
    }

    void validate(argagg::parser_results args)
    {
        if (!bool(args["input"]))
        {
            cerr << "Invalid usage! Input is missing!" << endl;
            cerr << dnn::input::parser;
            return;
        }

        if (!bool(args["net-input"]))
        {
            cerr << "Invalid usage! Network input is missing!" << endl;
            cerr << dnn::input::parser;
            return;
        }

        string str(args["input"].as<string>());

        auto images = utils::load_dataset(str);

        auto compatible_rect = rectangle(images[0].nc() + images[0].nc() % SR_SCALE, images[0].nr() + images[0].nr() % SR_SCALE);
        images = utils::resize_dataset(images, compatible_rect);
        auto valid = utils::resize_dataset(images, 1.0f / SR_SCALE);

        sr_net dnnet;
        deserialize(args["net-input"].as<string>()) >> dnnet;

        if (is_same<pixel_type, float>())
            utils::norm_dataset(valid, 1.0 / 255.0);

        auto res = dnnet.process_batch(valid, 1);

        if (is_same<pixel_type, float>())
            utils::norm_dataset(res, 255.0);

        float error = 0.0f;
        for (auto i = 0; i < images.size(); i++)
        {
            error += utils::square_difference(images[i], res[i]);
        }

        error /= images[0].nr() * images[0].nc() * images.size();

        cout << "The average pixel squared error on the whole validation set is: " << error << endl;
    }

    void evaluate(argagg::parser_results args)
    {
        if (!bool(args["input"]))
        {
            cerr << "Invalid usage! Input is missing!" << endl;
            cerr << dnn::input::parser;
            return;
        }

        if (!bool(args["net-input"]))
        {
            cerr << "Invalid usage! Network input is missing!" << endl;
            cerr << dnn::input::parser;
            return;
        }

        matrix<rgb_pixel> img;
        string str(args["input"].as<string>());
        load_image(img, str);

        matrix<pixel_type> img_gray;
        assign_image(img_gray, img);
        if (is_same<pixel_type, float>())
            utils::norm_image(img_gray, 1.0 / 255.0); // not needed if pixel_type is rgb_pixel

        std::vector<matrix<pixel_type>> eval;
        eval.push_back(img_gray);

        sr_net dnnet;
        deserialize(args["net-input"].as<string>()) >> dnnet;

        auto res = dnnet.process_batch(eval, 1);

        if (args["output"])
        {
            if (is_same<pixel_type, float>())
            {
                utils::norm_image(res[0], 255.0);
                utils::norm_image(eval[0], 255.0);
            }

            save_jpeg(res[0], args["output"].as<string>() + "_output.jpg");
            save_jpeg(eval[0], args["output"].as<string>() +"_eval.jpg");
        }

        //auto vifp_grade = quality::vifp(compatible_image, res[0]);
        //cout << "The VIFP index of evaluated image is: " << vifp_grade << endl;

#ifdef WIN32
        if (args["show"])
        {
            image_window original, net_output, difference;

            matrix<rgb_pixel> resized_img(img.nr() * SR_SCALE, img.nc() * SR_SCALE);
            resize_image(img, resized_img, interpolate_bilinear());

            original.set_image(resized_img);
            original.set_title("Original Image");

            net_output.set_image(res[0]);
            net_output.set_title("Evaluated Image");//". VIFP index of the image: " + to_string(vifp_grade));

            difference.set_image(utils::difference(resized_img, res[0]));
            difference.set_title("Difference between images");

            original.show();
            net_output.show();
            difference.show();

            int showing;
            cin >> showing;
        }
#endif
    }
}

int main(int argc, char** argv) try
{
    auto args = dnn::input::parser.parse(argc, argv);

    if (args["help"])
        print_help();

    if (args["train"])
        training(args);

    if (args["valid"])
        validate(args);

    if (args["eval"])
        evaluate(args);

    return 0;
}
catch (exception& e)
{
    cerr << e.what() << endl;
    return 1;
}
