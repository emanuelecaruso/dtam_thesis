# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/emanuele/Scrivania/idea3Dreconstr/prova_dr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build

# Include any dependencies generated for this target.
include executables/CMakeFiles/test_synth_cuda.dir/depend.make

# Include the progress variables for this target.
include executables/CMakeFiles/test_synth_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include executables/CMakeFiles/test_synth_cuda.dir/flags.make

executables/CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o: executables/CMakeFiles/test_synth_cuda.dir/flags.make
executables/CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o: ../executables/test_synth_cuda.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object executables/CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables/test_synth_cuda.cpp

executables/CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables/test_synth_cuda.cpp > CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.i

executables/CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables/test_synth_cuda.cpp -o CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.s

executables/CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o: executables/CMakeFiles/test_synth_cuda.dir/flags.make
executables/CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o: ../src/dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object executables/CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dataset.cpp

executables/CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dataset.cpp > CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.i

executables/CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dataset.cpp -o CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.s

executables/CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o: executables/CMakeFiles/test_synth_cuda.dir/flags.make
executables/CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object executables/CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/camera.cpp

executables/CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/camera.cpp > CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.i

executables/CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/camera.cpp -o CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.s

executables/CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o: executables/CMakeFiles/test_synth_cuda.dir/flags.make
executables/CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o: ../src/image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object executables/CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/image.cpp

executables/CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/image.cpp > CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.i

executables/CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/image.cpp -o CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.s

executables/CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o: executables/CMakeFiles/test_synth_cuda.dir/flags.make
executables/CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o: ../src/dtam_cuda.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object executables/CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dtam_cuda.cpp

executables/CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dtam_cuda.cpp > CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.i

executables/CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dtam_cuda.cpp -o CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.s

executables/CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o: executables/CMakeFiles/test_synth_cuda.dir/flags.make
executables/CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object executables/CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/utils.cpp

executables/CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/utils.cpp > CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.i

executables/CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/utils.cpp -o CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.s

executables/CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o: executables/CMakeFiles/test_synth_cuda.dir/flags.make
executables/CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o: ../src/cuda_handler.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object executables/CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/cuda_handler.cu -o CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o

executables/CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

executables/CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_synth_cuda
test_synth_cuda_OBJECTS = \
"CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o"

# External object files for target test_synth_cuda
test_synth_cuda_EXTERNAL_OBJECTS =

executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/build.make
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o: executables/CMakeFiles/test_synth_cuda.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CUDA device code CMakeFiles/test_synth_cuda.dir/cmake_device_link.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_synth_cuda.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
executables/CMakeFiles/test_synth_cuda.dir/build: executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o

.PHONY : executables/CMakeFiles/test_synth_cuda.dir/build

# Object files for target test_synth_cuda
test_synth_cuda_OBJECTS = \
"CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o" \
"CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o"

# External object files for target test_synth_cuda
test_synth_cuda_EXTERNAL_OBJECTS =

executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/test_synth_cuda.cpp.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/__/src/dataset.cpp.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/__/src/camera.cpp.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/__/src/image.cpp.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/__/src/dtam_cuda.cpp.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/__/src/utils.cpp.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/__/src/cuda_handler.cu.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/build.make
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
executables/test_synth_cuda: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/cmake_device_link.o
executables/test_synth_cuda: executables/CMakeFiles/test_synth_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable test_synth_cuda"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_synth_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
executables/CMakeFiles/test_synth_cuda.dir/build: executables/test_synth_cuda

.PHONY : executables/CMakeFiles/test_synth_cuda.dir/build

executables/CMakeFiles/test_synth_cuda.dir/clean:
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && $(CMAKE_COMMAND) -P CMakeFiles/test_synth_cuda.dir/cmake_clean.cmake
.PHONY : executables/CMakeFiles/test_synth_cuda.dir/clean

executables/CMakeFiles/test_synth_cuda.dir/depend:
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/emanuele/Scrivania/idea3Dreconstr/prova_dr /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables/CMakeFiles/test_synth_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : executables/CMakeFiles/test_synth_cuda.dir/depend

