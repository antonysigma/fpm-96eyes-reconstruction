#include <armadillo>
#include <catch2/catch_test_macros.hpp>
#include <highfive/H5File.hpp>

// Patch to encode std::complex<float> in HDF5 file.
#include "complex_float_support.hpp"
#include "constants.hpp"
#include "fpm-epry-runtime.h"
#include "read-slice.h"

using namespace arma;
using namespace HighFive;
using constants::tile_size;
using Halide::Runtime::Buffer;
using reconstruction::ComplexBuffer;

namespace {
constexpr auto n_illuminations = 25;
constexpr char filename[]{HDF5_FILE_PATH};
constexpr int well_id = 5;

/** Low-resolution frames, sorted by lateral LED offsets from the optical axis.
 * */
constexpr std::array low_res_frame_id{0,  5,  1,  6,  2,  7,  3,  8,  4,  13, 14, 9,  15,
                                      16, 10, 17, 18, 11, 19, 20, 12, 29, 37, 21, 38, 30,
                                      25, 31, 39, 22, 40, 32, 26, 33, 41, 23, 42, 34, 27,
                                      35, 43, 24, 44, 36, 28, 45, 46, 47, 48};
static_assert(low_res_frame_id.size() >= n_illuminations);

std::vector<size_t>
getFirstNFrameId(const size_t n) {
    std::vector<size_t> frame_id(n);
    std::copy_n(low_res_frame_id.begin(), n, frame_id.begin());
    return frame_id;
}

std::vector<size_t>
getGenericFrameId(const size_t n) {
    std::vector<size_t> frame_id(n);
    std::iota(frame_id.begin(), frame_id.end(), 0);
    return frame_id;
}

template <typename T>
struct coord_t {
    T kx{};
    T ky{};
};
static_assert(sizeof(coord_t<int32_t>) == sizeof(int32_t) * 2);

using C = coord_t<int32_t>;
constexpr std::array k_offset{
    C{128, 128}, C{108, 108}, C{128, 108}, C{147, 108}, C{147, 128}, C{147, 147}, C{128, 147},
    C{108, 147}, C{108, 128}, C{89, 108},  C{108, 89},  C{128, 89},  C{147, 89},  C{166, 108},
    C{166, 128}, C{166, 147}, C{147, 166}, C{128, 166}, C{108, 166}, C{89, 147},  C{89, 128},
    C{71, 109},  C{72, 90},   C{90, 90},   C{90, 72},   C{109, 71},  C{128, 71},  C{146, 71},
    C{165, 72},  C{165, 90},  C{183, 90},  C{184, 109}, C{184, 128}, C{184, 146}, C{183, 165},
    C{165, 165}, C{165, 183}, C{146, 184}, C{128, 184}, C{109, 184}, C{90, 183},  C{90, 165},
    C{72, 165},  C{71, 146},  C{71, 128},  C{73, 73},   C{182, 73},  C{182, 182}, C{73, 182}};
static_assert(k_offset.size() >= n_illuminations);

Mat<uint8_t>
stretchContrast(fmat input) {
    const float vmin = input.min();
    const float vmax = input.max();
    Mat<uint8_t> stretched = conv_to<Mat<uint8_t>>::from(  //
        (input - vmin) * 255.0f / (vmax - vmin + 1e-12f));

    return stretched;
}
}  // namespace

SCENARIO("Can run EPRY algorithm smoothly") {
    GIVEN("Raw data") {
        auto [raw, pupil] = []() -> std::pair<storage::u8_cube_t, ComplexBuffer> {
            // Mount HDF5 file
            auto file = File(filename, File::ReadOnly);

            // Mount dataset
            auto dataset = file.getDataSet("imlow");

            // Read raw low resolution images at the center of the camera view.
            const storage::roi_t center_roi{2592 / 2 - tile_size / 2, 1944 / 2 - tile_size / 2,
                                            tile_size};
            auto raw = storage::readFPMRaw(file.getDataSet("imlow"), well_id, center_roi,
                                           getFirstNFrameId(n_illuminations));

            {
                Cube<uint8_t> raw_buffer{raw.data(),      tile_size, tile_size,
                                         n_illuminations, false,     true};
                Mat<uint8_t>(trans(raw_buffer.slice(0))).save("small-roi-raw.pgm", pgm_binary);
            }

            // Read the pupil function initial guess
            ComplexBuffer pupil{2, tile_size, tile_size};
            file.getDataSet("initial_pupil")  //
                .select({well_id, 0, 0}, {1, tile_size, tile_size})
                .read(reinterpret_cast<std::complex<float>*>(pupil.data()));

            {
                cx_fmat pupil_buffer{reinterpret_cast<cx_float*>(pupil.data()), tile_size,
                                     tile_size, false, true};
                stretchContrast(trans(imag(pupil_buffer)))
                    .save("small-roi-initial-pupil.pgm", pgm_binary);
            }

            return {raw, pupil};
        }();

        Mat<int32_t> k_offset_buffer{&(k_offset.front().kx), 2, n_illuminations};

        WHEN("Initialize FPMEpryRunner") {
            reconstruction::FPMEpryRunner runner{std::move(k_offset_buffer), std::move(pupil),
                                                 std::move(raw)};
            REQUIRE(runner.n_illuminations == n_illuminations);

            THEN("Can reconstruct images") {
                runner.reconstruct(5);

                AND_THEN("Can retrieve new pupil and high-res image") {
                    const auto new_pupil = runner.downloadPupil();
                    stretchContrast(trans(imag(new_pupil)))
                        .save("small-roi-new_pupil.pgm", pgm_binary);
                }

                AND_THEN("Can download high res image") {
                    const auto high_res = runner.computeHighRes();
                    stretchContrast(trans(real(high_res)))
                        .save("small-roi-magnitude.pgm", pgm_binary);
                    stretchContrast(trans(imag(high_res))).save("small-roi-phase.pgm", pgm_binary);

                    REQUIRE(real(high_res).is_finite());
                    REQUIRE(imag(high_res).is_finite());
                }
            }
        }
    }
}
