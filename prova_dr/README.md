    PICP: Projective ICP

### Dependencies (Ubuntu 18.04 LTS)
The following packages are required <br>
Before installing anything one should make sure what packages are already installed on the system!

CMake 3.15:

    install CMake 3.15 https://cmake.org/files/v3.15/

Eigen3 stable version 3.3.9:

    http://eigen.tuxfamily.org, download stable version 3.3.9 and install it with cmake

CUDA toolkit version 9.1

    sudo apt install nvidia-cuda-toolkit=9.1.85-3ubuntu1

OpenCV version 4.2:

    installation guide here: https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7

  	Disable CUDNN by removing these lines:

        -D CUDA_ARCH_BIN=7.5 \
				-D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \

    gcc and g++ version must be 6.5 for this installation

  	be sure that the path to opencv_contrib-4.5.2/modules is correct

    be sure that CUDA toolkit is already installed (previous step)

### Compilation
From the system console, execute the build sequence (out of source build):

    mkdir build
    cd build
    cmake ..
    make

### Execution
The project provides the following 5 binaries in the `build/executables` folder:
- `./camera_test`: Testing of the pinhole camera projection function
- `./distance_map_test`: Testing of the 2d distance map computation (assuming projected points)
- `./correspondence_finder_test`: Unittest for the correspondence finding algorithm that uses the projection and the distance map functionality)
- `./picp_solver_test`: Least squares solver testing on artifical data and known correspondences
- `./picp_complete_test`: Complete PICP program
