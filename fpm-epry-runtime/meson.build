extern_cufft_lib = static_library('extern_cufft',
    sources: 'src/extern_cufft.cpp',
    #cpp_args: [
    #    '-DHALIDE_EXTERN_DEBUG',
    #],
    dependencies: [
        halide_runtime_dep,
        cufft_wrapper_dep,
    ],
)

fpm_epry_runtime_lib = library('fpm-epry-runtime',
    sources: [
        'src/fpm-epry-runtime.cpp',
        halide_generated_bin['low_res_init'],
        halide_generated_bin['high_res_init'],
        halide_generated_bin['high_res_restore'],
        halide_generated_bin['fpm_epry'],
    ],
    #gnu_symbol_visibility: 'hidden',
    include_directories: [
        'inc',
        common_inc,
    ],
    link_with: extern_cufft_lib,
    dependencies: [
        armadillo_dep,
        halide_runtime_dep,
        cufft_wrapper_dep,
    ],
)

fpm_epry_runner_smoke_test_exe = executable(
    'fpm-epry-runner-smoke-test',
    sources: [
        'tests/fpm-runtime-smoke-test.cpp',
        'tests/test_high_res_init.cpp',
        'tests/test_high_res_restore.cpp',
        halide_generated_bin['high_res_init'],
        halide_generated_bin['high_res_restore'],
    ],
    include_directories: [
        'inc',
        common_inc,
    ],
    link_with: fpm_epry_runtime_lib,
    dependencies: [
        catch2_dep,
        halide_runtime_dep,
        armadillo_dep,
    ],
)

test_fpm_epry_from_data_exe = executable(
    'test-fpm-epry-from-data',
    sources: 'tests/test-fpm-epry-from-data.cpp',
    include_directories: [
        'inc',
        common_inc,
    ],
    cpp_args: [
        '-DHDF5_FILE_PATH="@0@"'.format(datafile_path),
    ],
    link_with: fpm_epry_runtime_lib,
    dependencies: [
        catch2_dep,
        halide_runtime_dep,
        armadillo_dep,
        read_slice_dep,
    ],
)

test('Zeropadded forward FFT', fpm_epry_runner_smoke_test_exe,
    args: ['-r', 'tap', '[high_res_init]'],
    suite: 'epry',
    protocol: 'tap',
)

test('FPM-EPRY smoke test', fpm_epry_runner_smoke_test_exe,
    args: ['-r', 'tap', '[runner]'],
    suite: 'epry',
    protocol: 'tap',
)

test('FPM-EPRY data validation', test_fpm_epry_from_data_exe,
    args: ['-r', 'tap'],
    suite: 'epry',
    protocol: 'tap',
)