cd build
cmake -D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr ..
make
cd ..
