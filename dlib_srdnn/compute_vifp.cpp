#include "compute_vifp.h"
#include <dlib/matrix/matrix_utilities.h>
#include <dlib/matrix/matrix_math_functions.h>
#include <dlib/image_transforms/image_pyramid.h>
#include <dlib/image_transforms/thresholding.h>
#include <dlib/image_transforms.h>

using namespace dlib;

namespace dnn
{
namespace quality
{
    using namespace matrix_utility;

    namespace
    {
        const float SIGMA_NSQ = 2.0f;
        const int NLEVS = 4;
        const float EPSILON = 1e-10f;

        void gaussian(const matrix<float>& src, matrix<float>& dst, int ksize, double sigma)
        {
            int invalid = (ksize - 1) / 2;
            matrix<float> tmp(src.nr(), src.nc());
            gaussian_blur(src, tmp, sigma, ksize);
            dst = subm(tmp, range(invalid, tmp.nr() - invalid), range(invalid, tmp.nc() - invalid));
        }

        // OpenCV-like thresholding functions
        namespace
        {
            void threshold_binary(matrix<float>& mat, float thresh)
            {
                matrix<float> tmp(mat.nr(), mat.nc());
                threshold_image(mat, tmp, thresh);
                tmp /= 255.0;
                mat = tmp;
            }

            void threshold_tozero(matrix<float>& mat, float thresh)
            {
                matrix<float> tmp = mat;
                threshold_binary(tmp, thresh);
                mat = pointwise_multiply(mat, tmp);
            }

            void threshold_toeps(matrix<float>& mat, float thresh)
            {
                matrix<float> tmp1 = mat;
                matrix<float> tmp2 = mat * -1.f;
                threshold_tozero(tmp1, thresh);
                threshold_binary(tmp2, thresh);
                tmp2 *= EPSILON;
                mat = tmp1 + tmp2;
            }
        }

        void compute_vifp(matrix<float> ref, matrix<float> dist, int N, double& num, double& den)
        {
            int w = ref.nc() - (N - 1);
            int h = ref.nr() - (N - 1);

            matrix<float> tmp(h, w);
            matrix<float> mu1(h, w), mu2(h, w),
                mu1_sq(h, w), mu2_sq(h, w), mu1_mu2(h, w),
                sigma1_sq(h, w), sigma2_sq(h, w), sigma12(h, w), g(h, w), sv_sq(h, w);
            matrix<float> sigma1_sq_th, sigma2_sq_th, g_th;

            // mu1 = filter2(win, ref, 'valid');
            gaussian(ref, mu1, N, N / 5.0);
            // mu2 = filter2(win, dist, 'valid');
            gaussian(dist, mu2, N, N / 5.0);

            // mu1_sq = mu1.*mu1;
            mu1_sq = pointwise_multiply(mu1, mu1);
            // mu2_sq = mu2.*mu2;
            mu2_sq = pointwise_multiply(mu2, mu2);
            // mu1_mu2 = mu1.*mu2;
            mu1_mu2 = pointwise_multiply(mu1, mu2);

            // sigma1_sq = filter2(win, ref.*ref, 'valid') - mu1_sq;
            tmp = pointwise_multiply(ref, ref);
            gaussian(tmp, sigma1_sq, N, N / 5.0);
            sigma1_sq -= mu1_sq;
            // sigma2_sq = filter2(win, dist.*dist, 'valid') - mu2_sq;
            tmp = pointwise_multiply(dist, dist);
            gaussian(tmp, sigma2_sq, N, N / 5.0);
            sigma2_sq -= mu2_sq;
            // sigma12 = filter2(win, ref.*dist, 'valid') - mu1_mu2;
            tmp = pointwise_multiply(ref, dist);
            gaussian(tmp, sigma12, N, N / 5.0);
            sigma12 -= mu1_mu2;

            // sigma1_sq(sigma1_sq<0)=0;
            //threshold_image(sigma1_sq, tmp, 0.0f); // need to test this
            //sigma1_sq = pointwise_multiply(sigma1_sq, tmp); // TODO: multiply by 255, threshold it, divide by 255 (maybe this approach is better)
            threshold_tozero(sigma1_sq, 0.0f);
            // sigma2_sq(sigma2_sq<0)=0;
            threshold_tozero(sigma2_sq, 0.0f);

            // g=sigma12./(sigma1_sq+1e-10);
            tmp = sigma1_sq + EPSILON;
            g = pointwise_divide(sigma12, tmp);

            // sv_sq=sigma2_sq-g.*sigma12;
            tmp = pointwise_multiply(g, sigma12);
            sv_sq = sigma2_sq - tmp;

            // g(sigma1_sq<1e-10)=0;
            sigma1_sq_th = sigma1_sq;
            threshold_binary(sigma1_sq_th, EPSILON);
            g = pointwise_multiply(sigma1_sq_th, g);

            // sv_sq(sigma1_sq<1e-10)=sigma2_sq(sigma1_sq<1e-10);
            sv_sq = pointwise_multiply(sv_sq, sigma1_sq_th);
            tmp = pointwise_multiply(sigma2_sq, 1.0f - sigma1_sq_th);
            sv_sq += tmp;

            // sigma1_sq(sigma1_sq<1e-10)=0;
            threshold_tozero(sigma1_sq, EPSILON);

            // g(sigma2_sq<1e-10)=0;
            sigma2_sq_th = sigma2_sq;
            threshold_binary(sigma2_sq_th, EPSILON);
            g = pointwise_multiply(g, sigma2_sq_th);

            // sv_sq(sigma2_sq<1e-10)=0;
            sv_sq = pointwise_multiply(sv_sq, sigma2_sq_th);

            // sv_sq(g<0)=sigma2_sq(g<0);
            g_th = g;
            threshold_binary(g_th, 0);
            sv_sq = pointwise_multiply(sv_sq, g_th);
            tmp = pointwise_multiply(sigma2_sq, 1.0f - g_th);
            sv_sq += tmp;

            // g(g<0)=0;
            threshold_tozero(g, 0.0f);

            // sv_sq(sv_sq<=1e-10)=1e-10;
            threshold_toeps(sv_sq, EPSILON);

            // num=num+sum(sum(log10(1+g.^2.*sigma1_sq./(sv_sq+sigma_nsq))));
            sv_sq += SIGMA_NSQ;
            g = pointwise_multiply(g, g);
            g = pointwise_multiply(g, sigma1_sq);
            tmp = pointwise_divide(g, sv_sq);
            tmp += 1.0f;
            tmp = log10(tmp);
            num += sum(tmp);

            // den=den+sum(sum(log10(1+sigma1_sq./sigma_nsq)));
            tmp = 1.0f + sigma1_sq / SIGMA_NSQ;
            tmp = log10(tmp);
            den += sum(tmp);
        }
    }

    double vifp(const matrix<pixel_type>& original, const matrix<pixel_type>& processed)
    {
        matrix<float> orig, proc;

        assign_image(orig, original);

        if (original.size() != processed.size())
        {
            matrix<pixel_type> proc_resized(original.nr(), original.nc());
            resize_image(processed, proc_resized, interpolate_quadratic());
            assign_image(proc, proc_resized);
        }
        else
        {
            assign_image(proc, processed);
        }

        double num = 0.0;
        double den = 0.0;

        matrix<float> ref[NLEVS];
        matrix<float> dist[NLEVS];
        matrix<float> tmp1, tmp2;

        int w = original.nc();
        int h = original.nr();

        // for scale=1:4
        for (int scale = 0; scale<NLEVS; scale++) {
            // N=2^(4-scale+1)+1;
            int N = (2 << (NLEVS - scale - 1)) + 1;

            if (scale == 0) {
                ref[scale] = orig;
                dist[scale] = proc;
            }
            else {
                // ref=filter2(win,ref,'valid');
                gaussian(ref[scale - 1], tmp1, N, N / 5.0);
                // dist=filter2(win,dist,'valid');
                gaussian(dist[scale - 1], tmp2, N, N / 5.0);

                w = (w - (N - 1)) / 2;
                h = (h - (N - 1)) / 2;

                ref[scale] = matrix<float>(h, w);
                dist[scale] = matrix<float>(h, w);

                // ref=ref(1:2:end,1:2:end);
                resize_image(tmp1, ref[scale], interpolate_nearest_neighbor());
                // dist=dist(1:2:end,1:2:end);
                resize_image(tmp2, dist[scale], interpolate_nearest_neighbor());
            }

            compute_vifp(ref[scale], dist[scale], N, num, den);
        }

        return num / den;
    }
}
}