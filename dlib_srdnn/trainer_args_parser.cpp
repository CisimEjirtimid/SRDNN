#include "trainer_args_parser.h"
#include <vector>

namespace dnn
{
namespace args
{
    namespace
    {
        std::vector<string> split(const string& str, char separator)
        {
            std::vector<string> strings;
            istringstream f(str);
            string s;
            while (std::getline(f, s, separator)) {
                strings.push_back(s);
            }

            return strings;
        }

        std::map<string, string> get_args(const string& str)
        {
            std::map<string, string> args;

            auto str_args = split(str, ':');

            for (auto i = 0; i < str_args.size(); i++)
            {
                auto arg = split(str_args[i], '=');

                if (arg.size() != 2)
                {
                    std::cerr << "Parsing error, unable to parse arguments." << std::endl;
                    return std::map<string, string>();
                }

                args.insert(pair<string, string>(arg[0], arg[1]));
            }

            return args;
        }
    }

    parser::parser(dnn_trainer<sr_net>& trainer)
        : trainer(trainer)
    {
        // Setting default parameters to DNN trainer.
        trainer.set_synchronization_file("sync_file", chrono::minutes(1));
        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.00001);
        trainer.set_mini_batch_size(1);
        trainer.set_iterations_without_progress_threshold(2000);
    }

    void parser::set_args(std::map<string, string>& args)
    {
        if (!args.empty())
        {
            for (auto it = args.begin(); it != args.end(); ++it)
            {
                switch (code(it->first))
                {
                case synchronization_file:
                    trainer.set_synchronization_file(it->second, chrono::minutes(1));
                    break;
                case learning_rate:
                    trainer.set_learning_rate(stod(it->second));
                    break;
                case min_learning_rate:
                    trainer.set_min_learning_rate(stod(it->second));
                    break;
                case mini_batch_size:
                    trainer.set_mini_batch_size(stoi(it->second));
                    break;
                case iterations_without_progress_threshold:
                    trainer.set_iterations_without_progress_threshold(stoi(it->second));
                    break;
                default:
                    break;
                }
            }
        }
    }

    bool parser::parse(const string& args)
    {
        auto arguments = get_args(args);

        if (arguments.size() == 0)
            return false;

        set_args(arguments);

        return true;
    }
}
}