#include "defs.h"
#include "dataset.h"
#include "camera.h"
#include "state.h"
#include "image.h"

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

  CameraVector camera_vector; // initialize vector containing pointers to camera objects for each pose
  ImageVector image_vector; // initialize vector containing pointers to image objects of dataset
  //
  Eigen::Vector3f p(0,0,0);
  Dataset* dataset = new Dataset(path_dataset); // pointer to object handler of the dataset
  State* state = new State(p); // pointer to object handler of the state of the LS problem
  //
  // state->generateWorldFrame();  // generate world frame with points
  // dataset->collectCameras(camera_vector);
  dataset->collectImages(image_vector);
  image_vector[0]->show();
  //
  // for (Camera* camera : camera_vector)
  // {
  //   camera->projectCps(state->cps_world_frame);
  //   camera->showPointsWithCircles();
  // }
  // for (Image<cv::Vec3b>* image : image_vector)
  //   image->show();
  cv::waitKey(0);

  // camera_vector[8]->projectCps(state->cp_vector);
  // camera_vector[8]->showImage();

  return 1;
}
