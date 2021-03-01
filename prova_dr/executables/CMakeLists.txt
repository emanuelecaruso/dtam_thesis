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

if (USE_CUDA)
  #synthetic simulator test with cuda
  add_executable(test_synth_cuda test_synth_cuda.cpp
  ../src/dataset.cpp
  ../src_cuda/camera_cpu.cu
  ../src/image.cpp
  ../src_cuda/renderer.cu
  ../src_cuda/environment.cu
  ../src_cuda/dtam_cuda.cu
  ../src/utils.cpp
  #../src_cuda/cuda_handler.cu
  ../src_cuda/camera_gpu.cu
  )
  target_link_libraries(test_synth_cuda opencv_highgui)
  target_compile_options(test_synth_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
endif()
