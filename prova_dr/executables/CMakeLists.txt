link_libraries(${OpenCV_LIBS})

##main
#add_executable(main main.cpp
#../src/dataset.cpp
#../src/camera.cpp
#../src/image.cpp
#../src/renderer.cpp
#../src/environment.cpp
#../src/dtam.cpp
#../src/utils.cpp
#)
#target_link_libraries(main opencv_highgui)

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
  #synthetic simulator test with cuda
  add_executable(test_synth_cuda test_synth_cuda.cpp
  ../src_cuda/camera_cpu.cu
  ../src/image.cpp
  ../src/camera.cpp
  ../src/renderer.cpp
  ../src_cuda/environment.cu
  ../src/environment.cpp
  ../src_cuda/dtam_cuda.cu
  ../src/utils.cpp
  ../src_cuda/camera_gpu.cu
  ../src_cuda/cuda_utils.cuh
  )
  target_link_libraries(test_synth_cuda opencv_highgui)
  target_compile_options(test_synth_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
  set_property(TARGET test_synth_cuda
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()
