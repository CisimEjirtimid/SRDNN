#include "loss_layer.h"

namespace dnn
{
    size_t loss_avg_::tensor_index(const dlib::tensor& t, long sample, long row, long column, long k)
    {
        // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
        return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
    }

    std::ostream& operator<<(std::ostream& out, const loss_avg_& item)
    {
        out << "loss_avg";
        return out;
    }

    void to_xml(const loss_avg_& item, std::ostream& out)
    {
        out << "<loss_avg/>";
    }

    void serialize(const loss_avg_& item, std::ostream& out)
    {
        dlib::serialize("loss_avg_", out);
    }

    void deserialize(loss_avg_& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version != "loss_avg_")
            throw dlib::serialization_error("Unexpected version found while deserializing dlib::loss_avg_.  Instead found " + version);
    }

    static size_t tensor_index(const dlib::tensor& t, long sample, long row, long column, long k)
    {
        
    }
}
