#include <HalideBuffer.h>

#include <armadillo>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "constants.hpp"
#include "high_res_init.h"

using namespace arma;
using Catch::Matchers::WithinRel;
using Halide::Runtime::Buffer;

namespace {
constexpr auto T = constants::tile_size;
constexpr auto T2 = T * 2;
constexpr auto n_illuminations = 3;
}  // namespace

SCENARIO("Zeropadded FFT is valid", "[high_res_init]") {
    GIVEN("Blank image") {
        fcube low_res(T, T, n_illuminations, fill::ones);

        WHEN("Compute forward FFT") {
            Buffer<float, 3> low_res_buffer{low_res.memptr(), T, T, n_illuminations};
            Buffer<float, 3> f_high_res{T2, T2, 2};

            // Fill impossible values.
            f_high_res.fill(datum::nan);

            low_res_buffer.set_host_dirty();
            const auto has_error = high_res_init(low_res_buffer, f_high_res);
            REQUIRE(has_error == 0);
            f_high_res.copy_to_host();

            THEN("All zeros except at the center") {
                const fmat imag_component{f_high_res.data() + T2 * T2, T2, T2, false, true};
                REQUIRE(imag_component.is_zero());

                fmat real_component{f_high_res.data(), T2, T2};
                REQUIRE(max(vectorise(real_component)) > 0.0f);
                REQUIRE(real_component(T, T) > 0.0f);
                REQUIRE_THAT(real_component(T, T), WithinRel(T * T, 1e-6f));

                real_component(T, T) = 0.0f;
                REQUIRE(real_component.is_zero());
            }
        }
    }
}