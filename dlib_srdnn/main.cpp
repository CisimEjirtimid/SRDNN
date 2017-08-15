#include <dlib/dnn.h>

#include "dnn_utils.h"
#include "loss_layer.h"

using namespace dlib;
using namespace std;
using namespace dnn;

#include <dlib/gui_core.h>
#include <dlib/gui_widgets.h>

using sr_net = loss_pixel<add_prev1<
    relu<con<3, 1, 1, 1, 1,
    relu<con<64, 3, 3, 1, 1,
    relu<con<64, 5, 5, 1, 1,
    relu<con<64, 7, 7, 1, 1,
    relu<con<64, 9, 9, 1, 1,
    relu<con<64, 11, 11, 1, 1,
    relu<con<64, 13, 13, 1, 1,
    relu<con<64, 15, 15, 1, 1,
    relu<con<64, 17, 17, 1, 1,
    relu<con<64, 19, 19, 1, 1,
    relu<con<64, 21, 21, 1, 1,
    relu<con<64, 23, 23, 1, 1,
    relu<con<64, 25, 25, 1, 1,
    relu<con<64, 27, 27, 1, 1,
    tag1<upsample<2, input_rgb_image>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using simple_sr_net = loss_pixel<relu<
    add_prev1<bn_con<con<3, 1, 1, 1, 1,
    relu<bn_con<con<64, 3, 3, 1, 1,
    tag1<upsample<2, input_rgb_image>>>>>>>>>>;

using simple_sr_net_eval = loss_pixel<relu<
    add_prev1<affine<con<3, 1, 1, 1, 1,
    relu<affine<con<64, 3, 3, 1, 1,
    tag1<upsample<2, input_rgb_image>>>>>>>>>>;

//int main(int argc, char** argv)
//{
//    if (argc != 2)
//    {
//        cout << "Give a folder as input. It should contain images for super-resolution deep learning process." << endl;
//        return 1;
//    }
//
//    string str(argv[1]);
//
//    auto images = utils::load_dataset(str);
//    images = utils::resize_dataset(images, rectangle(images[0].nr() + 1, images[0].nc()));
//    auto downsampled = utils::resize_dataset(images, 0.5);
//
//    simple_sr_net dnnet;
//
//    dnn_trainer<simple_sr_net> trainer(dnnet);
//    trainer.set_learning_rate(0.1);
//    trainer.set_min_learning_rate(0.0001);
//    trainer.set_mini_batch_size(10);
//    
//    trainer.be_verbose();
//
//    trainer.set_synchronization_file("srdnn_sync_residual", chrono::minutes(1));
//
//    trainer.set_iterations_without_progress_threshold(500);
//
//    trainer.train(downsampled, images);
//
//    dnnet.clean();
//    serialize("simple_sr_network.dat") << dnnet;
//
//    int end;
//    cin >> end;
//}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Give an image as input. It should be image to be processed by super-resolution deep learning." << endl;
        return 1;
    }

    matrix<rgb_pixel> img;
    string str(argv[1]);
    load_image(img, str);

    simple_sr_net_eval dnnet;
    deserialize("simple_sr_network.dat") >> dnnet;

    std::vector<matrix<rgb_pixel>> eval;
    eval.push_back(img);

    auto res = dnnet(eval);

    matrix<rgb_pixel> resized(img.nr() / 2 + 1, img.nc() / 2);
    resize_image(img, resized, interpolate_bilinear());

    image_window img1, img2;
    img1.set_image(resized);
    img2.set_image(res[0]);
    img1.show();
    img2.show();

    int end;
    cin >> end;
    return end;
}