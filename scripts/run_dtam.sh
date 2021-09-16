#Name of the .blend file inside "blender_scenes" directory
dataset_name='bunny_scene'

cd prova_dr

# ./build/executables/dataset_maker
./build/executables/test_synth_cuda ${dataset_name}

# cuda-memcheck ./build/executables/test_synth_cuda dataset_name
