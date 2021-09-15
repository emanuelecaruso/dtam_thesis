dataset_name='bunny_scene'

cd ./blender_scenes
# blender --background ./${dataset_name}.blend --python ./python_scripts/script.py CYCLES 640
blender --background ./${dataset_name}.blend --python ./python_scripts/script.py
