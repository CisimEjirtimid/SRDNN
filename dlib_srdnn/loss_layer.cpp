#include "loss_layer.h"

namespace dnn
{
    size_t loss_pixel_::tensor_index(const dlib::tensor& t, long sample, long row, long column, long k)
    {
        // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
        return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
    }

    std::ostream& operator<<(std::ostream& out, const loss_pixel_& item)
    {
        out << "loss_pixel";
        return out;
    }

    void to_xml(const loss_pixel_& item, std::ostream& out)
    {
        out << "<loss_pixel/>";
    }

    void serialize(const loss_pixel_& item, std::ostream& out)
    {
        dlib::serialize("loss_pixel_", out);
    }

    void deserialize(loss_pixel_& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version != "loss_pixel_")
            throw dlib::serialization_error("Unexpected version found while deserializing dlib::loss_avg_.  Instead found " + version);
    }
}
