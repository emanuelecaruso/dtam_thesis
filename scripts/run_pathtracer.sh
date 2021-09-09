dataset_name='bunny_scene'

cd ./pathtrace/scenes
blender --background ./${dataset_name}.blend --python ./python_scripts/script.py

../bin/yscenetrace yocto_dataset/${dataset_name}/${dataset_name}.json -o ../../prova_dr/dataset/${dataset_name}/.jpg -t path -s 30 -r 640
