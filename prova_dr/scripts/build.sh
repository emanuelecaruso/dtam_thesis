cd build
cmake -D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr ..
make -j8
cd ..
