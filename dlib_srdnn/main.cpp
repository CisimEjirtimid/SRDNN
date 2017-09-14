#include <dlib/dnn.h>

#include "input_parser.h"
#include "dnn_utils.h"
#include "dnn_setup.h"

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
        trainer.set_synchronization_file("sync_file", chrono::minutes(1));

        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.00001);
        trainer.set_mini_batch_size(1);
        trainer.set_iterations_without_progress_threshold(2000);

        trainer.be_verbose();

        trainer.train(downsampled, images);

        dnnet.clean();

        if (args["output"])
            serialize(args["output"].as<string>()) << dnnet;

        if (args["xml"])
            net_to_xml(dnnet, args["xml"].as<string>());
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
        dlib::assign_image(img_gray, img);
        utils::norm_image(img_gray, 1.0/255.0); // if this is done in loss layer, then it's unneeded here

        sr_net dnnet;
        deserialize(args["net-input"].as<string>()) >> dnnet;

        std::vector<matrix<pixel_type>> eval;
        eval.push_back(img_gray);

        auto res = dnnet(eval);

        if (args["output"])
        {
            utils::norm_image(res[0], 255.0);
            save_jpeg(res[0], "output.jpg");
            utils::norm_image(eval[0], 255.0);
            save_jpeg(eval[0], "eval.jpg");
        }

        
        if (args["show"])
        {
            image_window original, net_output, difference;

            matrix<rgb_pixel> resized_img(img.nr() * SR_SCALE, img.nc() * SR_SCALE);
            resize_image(img, resized_img, interpolate_bilinear());

            original.set_image(resized_img);
            original.set_title("Original Image");

            net_output.set_image(res[0]);
            net_output.set_title("Evaluated Image");

            difference.set_image(utils::difference(resized_img, res[0]));
            difference.set_title("Difference between images");

            original.show();
            net_output.show();
            difference.show();

            int showing;
            cin >> showing;
        }
        
    }
}

int main(int argc, char** argv) try
{
    auto args = dnn::input::parser.parse(argc, argv);

    if (args["help"])
        print_help();

    if (args["train"])
        training(args);

    if (args["eval"])
        evaluate(args);

    return 0;
}
catch (exception& e)
{
    cerr << e.what() << endl;
    return 1;
}
