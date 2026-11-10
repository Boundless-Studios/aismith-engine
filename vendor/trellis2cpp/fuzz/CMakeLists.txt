# libFuzzer harnesses for the untrusted (non-GGUF) input paths.
# Requires clang; enable with the fuzz build:
#   CC=clang CXX=clang++ cmake -B build-fuzz -DTRELLIS2_FUZZ=ON
#   cmake --build build-fuzz -j
#   ./build-fuzz/fuzz/fuzz_image     corpus_image/     -max_len=1048576
#   ./build-fuzz/fuzz/fuzz_dinodata  corpus_dinodata/  -max_len=1048576

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "TRELLIS2_FUZZ requires clang (libFuzzer)")
endif()

foreach(target fuzz_image fuzz_dinodata)
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE trellis2)
    target_compile_options(${target} PRIVATE -fsanitize=fuzzer)
    target_link_options(${target} PRIVATE -fsanitize=fuzzer)
    target_compile_features(${target} PRIVATE cxx_std_14)
endforeach()
