#pragma once
#include "camera.h"
#include "image.h"
#include <cuda_runtime.h>

__global__ void CostVolumeMin_kernel(cv::cuda::PtrStepSzf dOutput);
__host__ void CostVolumeMin(CameraVector camera_vector);
