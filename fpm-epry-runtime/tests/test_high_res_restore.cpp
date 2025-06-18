#include <HalideBuffer.h>

#include <armadillo>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "constants.hpp"
#include "high_res_restore.h"

using namespace arma;
using Catch::Matchers::WithinRel;
using Halide::Runtime::Buffer;

namespace {
constexpr auto T = constants::tile_size;
constexpr auto T2 = T * 2;
constexpr auto n_illuminations = 3;
}  // namespace

SCENARIO("Cropped inverse FFT is valid", "[high_res_init]") {
    GIVEN("Fourier spectrum") {
        fcube f_high_res(T2, T2, 2, fill::zeros);
        f_high_res(T, T, 0) = T * T;

        WHEN("Compute forward FFT") {
            Buffer<float, 3> f_high_res_buffer{f_high_res.memptr(), T2, T2, 2};
            Buffer<float, 3> high_res{2, T, T};

            // Fill impossible values.
            high_res.fill(datum::nan);

            f_high_res_buffer.set_host_dirty();
            const auto has_error = high_res_restore(f_high_res_buffer, high_res);
            REQUIRE(has_error == 0);
            high_res.copy_to_host();

            THEN("All zeros except at the center") {
                const cx_fmat high_res_buffer{reinterpret_cast<cx_float*>(high_res.data()), T, T,
                                              false, true};

                REQUIRE(imag(high_res_buffer).is_zero());
                REQUIRE(norm(real(high_res_buffer) - 1.0f, "inf") <= 1e-6f);
            }
        }
    }
}