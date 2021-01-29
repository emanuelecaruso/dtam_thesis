link_libraries(${OpenCV_LIBS})

#main
add_executable(main main.cpp
../src/dataset.cpp
../src/camera.cpp
../src/image.cpp
../src/dtam.cpp
../src/utils.cpp
)
target_link_libraries(main opencv_highgui)

#synthetic simulator test
add_executable(test_synth test_synth.cpp
../src/dataset.cpp
../src/camera.cpp
../src/image.cpp
../src/dtam.cpp
../src/utils.cpp
)
target_link_libraries(test_synth opencv_highgui)
