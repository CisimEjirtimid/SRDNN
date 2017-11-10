#pragma once
#include <dlib/pixel.h>
#include <dlib/matrix/matrix.h>
#include <dlib/matrix/matrix_utilities.h>

namespace quality
{
    namespace matrix_utility
    {
        template <typename M1, typename M2>
        struct op_pointwise_divide : dlib::basic_op_mm<M1, M2>
        {
            op_pointwise_divide(const M1& m1_, const M2& m2_) : dlib::basic_op_mm<M1, M2>(m1_, m2_) {}

            typedef typename dlib::impl::compatible<typename M1::type, typename M2::type>::type type;
            typedef const type const_ret_type;
            const static long cost = M1::cost + M2::cost + 1;

            const_ret_type apply(long r, long c) const
            {
                return this->m1(r, c)/this->m2(r, c);
            }
        };

        template <
            typename EXP1,
            typename EXP2
        >
        dlib::matrix_op<op_pointwise_divide<EXP1, EXP2> > pointwise_divide(
                const dlib::matrix_exp<EXP1>& a,
                const dlib::matrix_exp<EXP2>& b
            )
        {
            COMPILE_TIME_ASSERT((dlib::impl::compatible<typename EXP1::type, typename EXP2::type>::value == true));
            COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
            COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NC == 0 || EXP2::NC == 0);
            DLIB_ASSERT(a.nr() == b.nr() &&
                a.nc() == b.nc(),
                "\tconst matrix_exp pointwise_divide(const matrix_exp& a, const matrix_exp& b)"
                << "\n\tYou can only make a do a pointwise divide with two equally sized matrices"
                << "\n\ta.nr(): " << a.nr()
                << "\n\ta.nc(): " << a.nc()
                << "\n\tb.nr(): " << b.nr()
                << "\n\tb.nc(): " << b.nc()
            );
            using op = op_pointwise_divide<EXP1, EXP2>;
            return matrix_op<op>(op(a.ref(), b.ref()));
        }
    }

    double vifp(dlib::matrix<dlib::rgb_pixel> ref_image, dlib::matrix<dlib::rgb_pixel> dist_image);
}
