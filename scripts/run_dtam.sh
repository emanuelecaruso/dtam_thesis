#Name of the .blend file inside "blender_scenes" directory

dataset_name='bunny_scene'
# dataset_name='classroom'
# dataset_name='sin_9cams'
# dataset_name='rotatedcube_25cams'

cd prova_dr

# ./build/executables/dataset_maker
./build/executables/test_tracking ${dataset_name}
# ./build/executables/test_mapping ${dataset_name}

# cuda-memcheck ./build/executables/test_synth_cuda ${dataset_name}
