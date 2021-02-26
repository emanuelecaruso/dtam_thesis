#include "defs.h"
#include "dataset.h"
#include "camera.h"
#include "image.h"
#include "renderer.h"
#include "utils.h"
#pragma diag_suppress 2739

using namespace std;
using namespace pr;


int main (int argc, char * argv[]) {

  std::string path_dataset = "";

  if (argc>1)
    path_dataset=argv[1];
  else
  {
    cout << "run: './main path_to_dataset_folder'" << endl;
    return 0;
  }

  float max_depth;  // max depth in depth map
  CameraVector camera_vector; // initialize vector containing pointers to camera objects for each pose
  ImageVector image_vector; // initialize vector containing pointers to image objects of dataset
  //
  Eigen::Vector3f p(0,0,0);
  Dataset* dataset = new Dataset(path_dataset); // pointer to object handler of the dataset
  //
  dataset->collectCameras(camera_vector, max_depth);
  dataset->collectImages(image_vector);

  Eigen::Vector3f o(0,0,0);

  for (Camera* camera : camera_vector)
  {
    // camera->showWorldFrame(o,0.01,20);
    camera->image_rgb_->show();
  }
  for (Image<cv::Vec3b>* image : image_vector)
  {
    image->show();
  }

  // camera_vector[8]->showWorldFrame(o,0.01,20);
  // camera_vector[8]->image_rgb_->show();

  cv::waitKey(0);

  return 1;
}
