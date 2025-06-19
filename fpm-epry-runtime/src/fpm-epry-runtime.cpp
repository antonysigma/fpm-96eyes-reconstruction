#include "fpm-epry-runtime.h"

#include <cassert>

#include "constants.hpp"
#include "fpm_epry.h"
#include "high_res_init.h"
#include "high_res_restore.h"
#include "low_res_init.h"

using arma::cx_float;
using arma::cx_fmat;
using arma::fcube;

namespace reconstruction {

using constants::tile_size;

FPMEpryRunner::FPMEpryRunner(arma::Mat<int32_t> _k_offset, ComplexBuffer p, Buffer<uint8_t, 3> raw,
                             const float gamma)
    : n_illuminations{static_cast<int32_t>(_k_offset.n_cols)},
      k_offset{std::move(_k_offset)},
      low_res{tile_size, tile_size, n_illuminations},
      f_high_res{tile_size * 2, tile_size * 2, 2},
      pupil{std::move(p)} {
    assert(k_offset.n_rows == 2);

    assert(raw.width() == tile_size);
    assert(raw.height() == tile_size);
    assert(raw.dim(2).extent() == n_illuminations);

    assert(pupil.dim(0).extent() == 2);
    assert(pupil.dim(1).extent() == tile_size);
    assert(pupil.dim(2).extent() == tile_size);

    pupil.set_host_dirty();
    raw.set_host_dirty();
    {
        const auto has_error = low_res_init(raw, gamma, low_res);
        assert(!has_error);
    }
    low_res.set_host_dirty();
    {
        const auto has_error = high_res_init(low_res, f_high_res);
        assert(!has_error);
    }
    f_high_res.device_sync();
}

FPMEpryRunner::FPMEpryRunner(FPMEpryRunner&& prev, arma::Mat<int32_t> k_offset,
                             Buffer<uint8_t, 3> raw, const float gamma)
    : n_illuminations{prev.n_illuminations},
      k_offset{std::move(k_offset)},
      low_res{std::move(prev.low_res)},
      f_high_res{std::move(prev.f_high_res)},
      pupil{std::move(prev.pupil)} {
    assert(raw.width() == tile_size);
    assert(raw.height() == tile_size);
    assert(raw.dim(2).extent() == n_illuminations);

    raw.set_host_dirty();
    {
        const auto has_error = low_res_init(raw, gamma, low_res);
        assert(!has_error);
    }
    low_res.device_sync();
}

void
FPMEpryRunner::reconstruct(size_t max_iter, bool is_blocking) {
    // Close the loop by setting the input and output buffers to be the same.
    auto& f_high_res_new = f_high_res;
    auto& pupil_new = pupil;

    Buffer<const int32_t, 2> k_offset_buffer{k_offset.memptr(), 2, n_illuminations};
    k_offset_buffer.set_host_dirty();

    for (size_t iter = 0; iter < max_iter; iter++) {
        const auto has_error =
            fpm_epry(low_res, f_high_res, pupil, k_offset_buffer, f_high_res_new, pupil_new);
        assert(!has_error);
    }

    // Now, wait for the algorithm to finish, and then copy the data from GPU to
    // CPU.
    if (is_blocking) {
        f_high_res.device_sync();
    }
}

arma::cx_fmat
FPMEpryRunner::computeHighRes() {
    arma::cx_fmat high_res(tile_size, tile_size);

    Halide::Runtime::Buffer<float, 3> high_res_buffer{reinterpret_cast<float*>(high_res.memptr()),
                                                      2, tile_size, tile_size};

    // Apply inverse 2D fourier transform.
    const auto has_error = high_res_restore(f_high_res, high_res_buffer);
    assert(!has_error);

    high_res_buffer.copy_to_host();
    return high_res;
}

cx_fmat
FPMEpryRunner::downloadFourierPlane() {
    f_high_res.copy_to_host();
    const fcube f_high_res_buffer{f_high_res.data(), tile_size * 2, tile_size * 2, 2, false, true};

    return cx_fmat{f_high_res_buffer.slice(0), f_high_res_buffer.slice(1)};
}

arma::cx_fmat
FPMEpryRunner::downloadPupil() {
    pupil.copy_to_host();
    return arma::cx_fmat{reinterpret_cast<cx_float*>(pupil.data()), tile_size, tile_size, false,
                         true};
}

}  // namespace reconstruction