import bpy
import os
import json
import shutil
import numpy as np
import math
import sys
from shutil import copy

scene=bpy.context.scene
scene_name=scene.name
render=scene.render

cwd = os.getcwd()

scene_path_dtam=cwd+"/../prova_dr/dataset/"+scene_name+"/"
json_path=scene_path_dtam+"state.json"


# Opening JSON file
f = open(json_path)

# returns JSON object as
# a dictionary
data = json.load(f)

# Iterating through the json
# list

size=0.004
coll_name="state_coll"
for coll in bpy.data.collections:
    if coll.name==coll_name:
        for obj in coll.objects:
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(coll)

coll=bpy.data.collections.new(coll_name)
scene.collection.children.link(coll)

dict = list(data['cameras'].values())[0]
i=0
for key, value in dict.items():
    #if(((i%10)==0) and ((int(i/480)%10)==0)):
    if(True):
        color=value['color']
        position=value['position']

        bpy.ops.mesh.primitive_cube_add(size=size, location=(position[0],position[1],position[2]))
        bpy.context.active_object.name = key
        obj_ = bpy.data.objects[key]

        # collection for the cloud of points
        obj_old_coll = obj_.users_collection
        coll.objects.link(obj_)
        for ob in obj_old_coll: #unlink from all  precedent obj collections
            ob.objects.unlink(obj_)
    i=i+1



# Closing file
f.close()
