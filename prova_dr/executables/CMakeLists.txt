link_libraries(${OpenCV_LIBS})

##synthetic simulator test
#add_executable(test_synth test_synth.cpp
#../src/dataset.cpp
#../src/camera.cpp
#../src/image.cpp
#../src/renderer.cpp
#../src/environment.cpp
#../src/dtam.cpp
#../src/utils.cpp
#)
#target_link_libraries(test_synth opencv_highgui)

#app for creating datasets
add_executable(dataset_maker dataset_maker.cpp
../src/camera.cpp
../src/image.cpp
../src/renderer.cpp
../src/environment.cpp
../src/dtam.cpp
../src/utils.cpp
)
target_link_libraries(dataset_maker opencv_highgui)

if (USE_CUDA)
#test mapping
add_executable(test_mapping test_mapping.cpp
../src_cuda/camera_cpu.cu
../src/image.cpp
../src/camera.cpp
../src/renderer.cpp
../src_cuda/environment.cu
../src/environment.cpp
../src_cuda/dtam_cuda.cu
../src/utils.cpp
../src_cuda/camera_gpu.cu
../src_cuda/mapper.cu
../src_cuda/cuda_utils.cuh
)
target_link_libraries(test_mapping opencv_highgui)
target_compile_options(test_mapping PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
set_property(TARGET test_mapping PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#test tracking
add_executable(test_tracking test_tracking.cpp
../src_cuda/camera_cpu.cu
../src/image.cpp
../src/camera.cpp
../src/renderer.cpp
../src_cuda/environment.cu
../src/environment.cpp
../src_cuda/dtam_cuda.cu
../src/utils.cpp
../src_cuda/camera_gpu.cu
../src_cuda/mapper.cu
../src_cuda/cuda_utils.cuh
)
target_link_libraries(test_tracking opencv_highgui)
target_compile_options(test_tracking PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
set_property(TARGET test_tracking PROPERTY CUDA_SEPARABLE_COMPILATION ON)

endif()
