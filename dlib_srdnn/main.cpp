#include <dlib/dnn.h>

#ifdef _DEBUG
#include <dlib/gui_core.h>
#include <dlib/gui_widgets.h>
#endif

#include "loss_layer.h"
#include "dnn_utils.h"

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

    auto images = utils::load_dataset(std::string(argv[1]));
    auto downsampled = utils::resize_dataset(images, 0.5);
    auto dnninput = utils::resize_dataset(downsampled, dlib::rectangle(images[0].nc(), images[0].nr()));

#ifdef _DEBUG
    dlib::image_window img1, img2;
    img1.set_image(images[0]);
    img2.set_image(dnninput[0]);
    img1.show();
    img2.show();

    int end;
    cin >> end;
#endif

    using sr_net = loss_avg<
        relu<con<16, 1, 1, 1, 1,
        relu<con<16, 3, 3, 1, 1,
        relu<con<16, 5, 5, 1, 1,
        relu<con<16, 7, 7, 1, 1,
        relu<con<16, 9, 9, 1, 1,
        input<matrix<rgb_pixel>>>>>>>>>>>>>;

    sr_net dnnet;

    dnn_trainer<sr_net> trainer(dnnet);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.0001);
    trainer.set_mini_batch_size(100);
    trainer.be_verbose();

    trainer.set_synchronization_file("srdnn_sync", std::chrono::seconds(60));

    trainer.train(downsampled, images);
}