    PICP: Projective ICP

### Dependencies (Ubuntu 18.04 LTS)
The following packages are required <br>
Before installing anything one should make sure what packages are already installed on the system!

CMake build utilities:

    sudo apt install build-essential cmake

Eigen3 version 3.3.9: http://eigen.tuxfamily.org

OpenCV version 4.2, installation guide here: https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
Be sure that the flag WITH_CUDA=ON during the installation with cmake is set.

CUDA toolkit version 9.1

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
