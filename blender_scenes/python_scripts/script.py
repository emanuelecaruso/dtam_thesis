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
json_path=scene_path_dtam+scene_name+".json"

rem = False
con = False

if os.path.isdir(scene_path_dtam):
    print("directory "+scene_path_dtam+" already exists")
    while(True):
        print("Enter:")
        print("r to remove existing dataset")
        print("c to continue the existing dataset")
        print("q to quit")
        input1 = input()
        if (input1=="r"):
            rem=True
            break
        elif (input1=="c"):
            con=True
            break
        elif (input1=="q"):
            sys.exit()

if(rem):
    shutil.rmtree(scene_path_dtam, ignore_errors=True)

scene.display_settings.display_device = 'sRGB'


#############################################################################
#		generate json for dtam and render images
#############################################################################

engine=''
argv=sys.argv;
if argv[-2]=='CYCLES':
    engine='CYCLES'
    bpy.context.scene.cycles.samples = int(argv[-1])
else:
    engine='BLENDER_EEVEE'

engine_eevee='BLENDER_EEVEE'

scene.use_nodes = True
tree = scene.node_tree
links = tree.links


data_ = {}
if( not con):
    data_['cameras']={}
else:
    with open(json_path, "r+") as file:
        data_ = json.load(file)

argv = sys.argv

for n in tree.nodes:
    tree.nodes.remove(n)

rl = tree.nodes.new('CompositorNodeRLayers')
v = tree.nodes.new('CompositorNodeComposite')
v.use_alpha = False


mm = tree.nodes.new('CompositorNodeMath')
mm.operation='MULTIPLY'
mm.inputs[1].default_value=2

links.new(rl.outputs['Depth'], mm.inputs[0])

mp = tree.nodes.new('CompositorNodeMath')
mp.operation='POWER'
mp.inputs[1].default_value=-1
links.new(mm.outputs[0], mp.inputs[0])


i=0
# iterate through objects
for obj_ in bpy.data.objects:
    if obj_.type=="CAMERA":


        name_=obj_.name
        name_rgb="rgb_"+name_+".png"
        name_depth="depth_"+name_+".png"

        path_rgb=scene_path_dtam+name_rgb
        path_depth=scene_path_dtam+name_depth

        camera=bpy.data.cameras[name_]
        scene.frame_current=i

        l1=links.new(rl.outputs['Image'], v.inputs[0])
        if( (con and not os.path.isfile(path_rgb)) or not con ):
            scene.view_settings.view_transform = 'Standard'
            render.engine=engine
            render.filepath = os.path.join(scene_path_dtam,name_rgb )
            bpy.ops.render.render(write_still = True)

        links.remove(l1)
        l2=links.new(mp.outputs[0], v.inputs[0])
        if( (con and not os.path.isfile(path_depth)) or not con ):
            scene.view_settings.view_transform = 'Raw'
            render.engine=engine_eevee
            render.filepath = os.path.join(scene_path_dtam,name_depth )
            bpy.ops.render.render(write_still = True)


        pi=math.pi

        eul_ang_=obj_.rotation_euler
        ex_=eul_ang_.x
        ey_=eul_ang_.y
        ez_=eul_ang_.z


        Rx_ = np.matrix([[1, 0, 0],[0, math.cos(ex_), -math.sin(ex_)],[0, math.sin(ex_), math.cos(ex_)]])
        Ry_ = np.matrix([[math.cos(ey_), 0, math.sin(ey_)],[0, 1, 0],[-math.sin(ey_), 0, math.cos(ey_)]])
        Rz_ = np.matrix([[math.cos(ez_), -math.sin(ez_), 0],[math.sin(ez_), math.cos(ez_), 0],[0, 0, 1]])


        # here euler angles are applyied in zyx order
        Rzy_=np.dot(Rz_,Ry_)
        R_=np.dot(Rzy_,Rx_)
        l_=obj_.location

        frame_=[ R_.item(0), R_.item(1), R_.item(2),
                R_.item(3), R_.item(4), R_.item(5),
                R_.item(6), R_.item(7), R_.item(8),
                l_[0], l_[1], l_[2] ]


        resolution_x=render.resolution_x
        resolution_y=render.resolution_y
        aspect=resolution_x/resolution_y
        lens=bpy.data.cameras[name_].lens/1000
        width=bpy.data.cameras[name_].sensor_width/1000

        data_['cameras'][name_]={}
        data_['cameras'][name_]['aspect']=aspect
        data_['cameras'][name_]['width']=width
        data_['cameras'][name_]['resolution']=resolution_x
        data_['cameras'][name_]['frame']=frame_
        data_['cameras'][name_]['lens']=lens


        with open(json_path, 'w') as outfile_:
            json.dump(data_, outfile_)

        i=i+1
