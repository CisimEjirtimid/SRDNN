#include <dlib/dnn.h>

#include "dnn_setup.h"
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

    std::string str(argv[1]);

    auto images = utils::load_dataset(str);
    auto downsampled = utils::resize_dataset(images, 0.5);

    sr_net dnnet;

    dnn_trainer<sr_net> trainer(dnnet);
    trainer.set_learning_rate(0.1);
    trainer.set_min_learning_rate(0.0001);
    trainer.set_mini_batch_size(1);
    trainer.be_verbose();

    trainer.set_synchronization_file("srdnn_sync_residual", chrono::minutes(10));

    trainer.train(downsampled, images);
}
