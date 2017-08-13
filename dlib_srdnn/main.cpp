#include <dlib/dnn.h>

#include "dnn_utils.h"
#include "loss_layer.h"

using namespace dlib;
using namespace std;
using namespace dnn;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Give a folder as input.  It should contain images for super-resolution deep learning process." << endl;
        return 1;
    }

    string str(argv[1]);

    auto images = utils::load_dataset(str);
    images = utils::resize_dataset(images, rectangle(images[0].nr() + 1, images[0].nc()));
    auto downsampled = utils::resize_dataset(images, 0.5);

    using sr_net = loss_avg<add_prev1<
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

    sr_net dnnet;

    dnn_trainer<sr_net> trainer(dnnet);
    trainer.set_learning_rate(0.1);
    trainer.set_min_learning_rate(0.0001);
    trainer.set_mini_batch_size(1);
    trainer.be_verbose();

    trainer.set_synchronization_file("srdnn_sync_residual", chrono::minutes(10));

    trainer.train(downsampled, images);
}
