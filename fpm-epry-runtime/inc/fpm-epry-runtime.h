#pragma once
#include <HalideBuffer.h>

#include <armadillo>

namespace reconstruction {

using ComplexBuffer = Halide::Runtime::Buffer<float, 3>;
using Halide::Runtime::Buffer;

class FPMEpryRunner {
   public:
    /** Initialize the high-resolution image by sinc interpolation.
     *
     * @param[in] gamma Gamma intensity correction of the raw pixels, plus 0.5.
     * (0.5 is equivalent to square root, used for conversion from intensity
     * value to amplitude value.)
     */
    FPMEpryRunner(arma::Mat<int32_t> k_offset, ComplexBuffer pupil, Buffer<uint8_t, 3> raw,
                  const float gamma = 0.6f);

    /** Parallax enhanced pupil recovery mode. */
    FPMEpryRunner(FPMEpryRunner&&, arma::Mat<int32_t> k_offset, Buffer<uint8_t, 3> raw,
                  const float gamma = 0.6f);

    /** Apply FPM-EPRY reconstuction. */
    void reconstruct(size_t max_iter = 20, bool blocking = true);

    /** Apply inverse Fourier transform and return the high-resolution image. */
    arma::cx_fmat computeHighRes();

    /** Download the pupil function. */
    arma::cx_fmat downloadPupil();

    /** Download the high-frequency Fourier spectrum. */
    arma::cx_fmat downloadFourierPlane();

    const int32_t n_illuminations;

   private:
    const arma::Mat<int32_t> k_offset;
    Buffer<float, 3> low_res;

    ComplexBuffer f_high_res;
    ComplexBuffer pupil;
};
}  // namespace reconstruction