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
include executables/CMakeFiles/test_synth.dir/depend.make

# Include the progress variables for this target.
include executables/CMakeFiles/test_synth.dir/progress.make

# Include the compile flags for this target's objects.
include executables/CMakeFiles/test_synth.dir/flags.make

executables/CMakeFiles/test_synth.dir/test_synth.cpp.o: executables/CMakeFiles/test_synth.dir/flags.make
executables/CMakeFiles/test_synth.dir/test_synth.cpp.o: ../executables/test_synth.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object executables/CMakeFiles/test_synth.dir/test_synth.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth.dir/test_synth.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables/test_synth.cpp

executables/CMakeFiles/test_synth.dir/test_synth.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth.dir/test_synth.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables/test_synth.cpp > CMakeFiles/test_synth.dir/test_synth.cpp.i

executables/CMakeFiles/test_synth.dir/test_synth.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth.dir/test_synth.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables/test_synth.cpp -o CMakeFiles/test_synth.dir/test_synth.cpp.s

executables/CMakeFiles/test_synth.dir/__/src/dataset.cpp.o: executables/CMakeFiles/test_synth.dir/flags.make
executables/CMakeFiles/test_synth.dir/__/src/dataset.cpp.o: ../src/dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object executables/CMakeFiles/test_synth.dir/__/src/dataset.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth.dir/__/src/dataset.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dataset.cpp

executables/CMakeFiles/test_synth.dir/__/src/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth.dir/__/src/dataset.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dataset.cpp > CMakeFiles/test_synth.dir/__/src/dataset.cpp.i

executables/CMakeFiles/test_synth.dir/__/src/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth.dir/__/src/dataset.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dataset.cpp -o CMakeFiles/test_synth.dir/__/src/dataset.cpp.s

executables/CMakeFiles/test_synth.dir/__/src/camera.cpp.o: executables/CMakeFiles/test_synth.dir/flags.make
executables/CMakeFiles/test_synth.dir/__/src/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object executables/CMakeFiles/test_synth.dir/__/src/camera.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth.dir/__/src/camera.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/camera.cpp

executables/CMakeFiles/test_synth.dir/__/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth.dir/__/src/camera.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/camera.cpp > CMakeFiles/test_synth.dir/__/src/camera.cpp.i

executables/CMakeFiles/test_synth.dir/__/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth.dir/__/src/camera.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/camera.cpp -o CMakeFiles/test_synth.dir/__/src/camera.cpp.s

executables/CMakeFiles/test_synth.dir/__/src/image.cpp.o: executables/CMakeFiles/test_synth.dir/flags.make
executables/CMakeFiles/test_synth.dir/__/src/image.cpp.o: ../src/image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object executables/CMakeFiles/test_synth.dir/__/src/image.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth.dir/__/src/image.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/image.cpp

executables/CMakeFiles/test_synth.dir/__/src/image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth.dir/__/src/image.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/image.cpp > CMakeFiles/test_synth.dir/__/src/image.cpp.i

executables/CMakeFiles/test_synth.dir/__/src/image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth.dir/__/src/image.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/image.cpp -o CMakeFiles/test_synth.dir/__/src/image.cpp.s

executables/CMakeFiles/test_synth.dir/__/src/dtam.cpp.o: executables/CMakeFiles/test_synth.dir/flags.make
executables/CMakeFiles/test_synth.dir/__/src/dtam.cpp.o: ../src/dtam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object executables/CMakeFiles/test_synth.dir/__/src/dtam.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth.dir/__/src/dtam.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dtam.cpp

executables/CMakeFiles/test_synth.dir/__/src/dtam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth.dir/__/src/dtam.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dtam.cpp > CMakeFiles/test_synth.dir/__/src/dtam.cpp.i

executables/CMakeFiles/test_synth.dir/__/src/dtam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth.dir/__/src/dtam.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/dtam.cpp -o CMakeFiles/test_synth.dir/__/src/dtam.cpp.s

executables/CMakeFiles/test_synth.dir/__/src/utils.cpp.o: executables/CMakeFiles/test_synth.dir/flags.make
executables/CMakeFiles/test_synth.dir/__/src/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object executables/CMakeFiles/test_synth.dir/__/src/utils.cpp.o"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_synth.dir/__/src/utils.cpp.o -c /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/utils.cpp

executables/CMakeFiles/test_synth.dir/__/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_synth.dir/__/src/utils.cpp.i"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/utils.cpp > CMakeFiles/test_synth.dir/__/src/utils.cpp.i

executables/CMakeFiles/test_synth.dir/__/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_synth.dir/__/src/utils.cpp.s"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/src/utils.cpp -o CMakeFiles/test_synth.dir/__/src/utils.cpp.s

# Object files for target test_synth
test_synth_OBJECTS = \
"CMakeFiles/test_synth.dir/test_synth.cpp.o" \
"CMakeFiles/test_synth.dir/__/src/dataset.cpp.o" \
"CMakeFiles/test_synth.dir/__/src/camera.cpp.o" \
"CMakeFiles/test_synth.dir/__/src/image.cpp.o" \
"CMakeFiles/test_synth.dir/__/src/dtam.cpp.o" \
"CMakeFiles/test_synth.dir/__/src/utils.cpp.o"

# External object files for target test_synth
test_synth_EXTERNAL_OBJECTS =

executables/test_synth: executables/CMakeFiles/test_synth.dir/test_synth.cpp.o
executables/test_synth: executables/CMakeFiles/test_synth.dir/__/src/dataset.cpp.o
executables/test_synth: executables/CMakeFiles/test_synth.dir/__/src/camera.cpp.o
executables/test_synth: executables/CMakeFiles/test_synth.dir/__/src/image.cpp.o
executables/test_synth: executables/CMakeFiles/test_synth.dir/__/src/dtam.cpp.o
executables/test_synth: executables/CMakeFiles/test_synth.dir/__/src/utils.cpp.o
executables/test_synth: executables/CMakeFiles/test_synth.dir/build.make
executables/test_synth: /usr/local/lib/libopencv_gapi.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_stitching.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_aruco.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_bgsegm.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_bioinspired.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_ccalib.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudabgsegm.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudafeatures2d.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudaobjdetect.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudastereo.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_dnn_objdetect.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_dnn_superres.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_dpm.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_face.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_freetype.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_fuzzy.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_hfs.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_img_hash.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_quality.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_reg.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_rgbd.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_saliency.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_stereo.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_structured_light.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_superres.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_surface_matching.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_tracking.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_videostab.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_xphoto.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_highgui.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_shape.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_datasets.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_plot.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_text.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_dnn.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_ml.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_videoio.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudaoptflow.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudalegacy.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudawarping.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_optflow.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_ximgproc.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_video.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_objdetect.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_calib3d.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_features2d.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_flann.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_photo.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudaimgproc.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudafilters.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_imgproc.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudaarithm.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_core.so.4.2.0
executables/test_synth: /usr/local/lib/libopencv_cudev.so.4.2.0
executables/test_synth: executables/CMakeFiles/test_synth.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable test_synth"
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_synth.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
executables/CMakeFiles/test_synth.dir/build: executables/test_synth

.PHONY : executables/CMakeFiles/test_synth.dir/build

executables/CMakeFiles/test_synth.dir/clean:
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables && $(CMAKE_COMMAND) -P CMakeFiles/test_synth.dir/cmake_clean.cmake
.PHONY : executables/CMakeFiles/test_synth.dir/clean

executables/CMakeFiles/test_synth.dir/depend:
	cd /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/emanuele/Scrivania/idea3Dreconstr/prova_dr /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/executables /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables /home/emanuele/Scrivania/idea3Dreconstr/prova_dr/build/executables/CMakeFiles/test_synth.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : executables/CMakeFiles/test_synth.dir/depend

