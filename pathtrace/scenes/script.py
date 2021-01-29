import bpy
import os 
import json
import shutil
import numpy as np
import math

scene_name=bpy.context.scene.name

cwd = os.getcwd()
scene_path=cwd+"/"+scene_name
shutil.rmtree(scene_name, ignore_errors=True)
os.mkdir(scene_path)
os.mkdir(scene_path+"/shapes/")

#############################################################################
#		generate json for path tracer
#############################################################################

data = {}
data['asset']={}
data['asset']['generator']="Yocto/GL - https://github.com/xelatihy/yocto-gl"
data['cameras']={}
data['materials']={}
data['objects']={}

# iterate through objects
for obj in bpy.data.objects:
    name=obj.name
    eul_ang=obj.rotation_euler
    ex=-eul_ang.x
    ey=-eul_ang.y
    ez=-eul_ang.z
    Rx = np.matrix([[1, 0, 0],[0, math.cos(ex), -math.sin(ex)],[0, math.sin(ex), math.cos(ex)]])
    Ry = np.matrix([[math.cos(ey), 0, math.sin(ey)],[0, 1, 0],[-math.sin(ey), 0, math.cos(ey)]])
    Rz = np.matrix([[math.cos(ez), -math.sin(ez), 0],[math.sin(ez), math.cos(ez), 0],[0, 0, 1]])
    Rxy=np.dot(Rx,Ry)
    R=np.dot(Rxy,Rz)

    l=obj.location

    frame=[ R.item(0), R.item(1), R.item(2),
            R.item(3), R.item(4), R.item(5),
            R.item(6), R.item(7), R.item(8),
            l[0], l[1], l[2] ]
    
    if obj.type=="CAMERA":
        lens=obj.data.lens*0.001
        data['cameras'][name]={}
        data['cameras'][name]['aspect']=1
        data['cameras'][name]['width']=0.024
        data['cameras'][name]['resolution']=720
        data['cameras'][name]['frame']=frame
        data['cameras'][name]['lens']=lens
    elif obj.type=="MESH":

        obj.select_set(True)
        bpy.ops.export_mesh.ply(
                filepath=scene_path+"/shapes/"+name+".ply",
                use_selection=True)
        obj.select_set(False)
        
        color = obj.color
        active_material=obj.active_material
        
        data['objects'][name]={}
        data['objects'][name]['material']=name
        data['objects'][name]['shape']=name
        
        data['materials'][name]={}
        
        if active_material is not None and 'Emission' in obj.active_material.node_tree.nodes:
            emission_strength = active_material.node_tree.nodes['Emission'].inputs[1].default_value
            emission_color = active_material.node_tree.nodes['Emission'].inputs[0].default_value
            er=emission_strength*emission_color[0]
            eg=emission_strength*emission_color[1]
            eb=emission_strength*emission_color[2]
            data['materials'][name]['displacement']=1
            data['materials'][name]['color']= [color[0],color[1],color[2]]
            data['materials'][name]['emission']=[er,eg,eb]
        else:
            data['materials'][name]['displacement']=1
            data['materials'][name]['color']=[color[0],color[1],color[2]]
        

with open(scene_path+"/"+scene_name+".json", 'w') as outfile:
    json.dump(data, outfile)



#############################################################################
#		generate json for LS
#############################################################################



data_ = {}
data_['cameras']={}

# iterate through objects
for obj_ in bpy.data.objects:
    if obj_.type=="CAMERA":
        
        name_=obj_.name
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
            
    
        lens_=obj_.data.lens*0.001 #millimiters to meters
        data_['cameras'][name_]={}
        data_['cameras'][name_]['aspect']=1
        data_['cameras'][name_]['width']=0.024
        data_['cameras'][name_]['resolution']=720
        data_['cameras'][name_]['frame']=frame_
        data_['cameras'][name_]['lens']=lens_
        
    
        

with open(cwd+"/../../prova_dr/dataset/"+scene_name+"/"+scene_name+".json", 'w') as outfile_:
    json.dump(data_, outfile_)
    

