import bpy
import os
import json
import shutil
import numpy as np
import math
import mathutils

scene=bpy.context.scene
render=scene.render

aspect=1.333333333333333
lens_meters=0.035
width_meters=0.024
resolution_x=640
resolution_y=resolution_x/aspect

print("resolution: "+str(resolution_x)+"x"+str(resolution_y)+"pxl")

lens_millimeters=lens_meters*1000
width_millimeters=width_meters*1000

render.resolution_x=resolution_x
render.resolution_y=resolution_y

cams_num=120
scene.frame_start=0
scene.frame_end=cams_num-1

coll_name="cams_coll"
for coll in bpy.data.collections:
    if coll.name==coll_name:
        bpy.data.collections.remove(coll)
        
coll=bpy.data.collections.new(coll_name)
scene.collection.children.link(coll)

bpy.data.scenes[0].timeline_markers.clear()
    
for cam in bpy.data.cameras:
    bpy.data.cameras.remove(cam)
    
for obj in bpy.data.objects:
    if obj.type=="CAMERA":
        bpy.data.objects.remove(obj)

for i in range(cams_num):
    num='{0:04}'.format(i)
    name_='Camera'+num
    camera_data = bpy.data.cameras.new(name='Camera'+num)
    camera_data.lens=lens_millimeters
    camera_data.sensor_width=width_millimeters
    camera_object = bpy.data.objects.new('Camera'+num, camera_data)
    bpy.context.scene.collection.objects.link(camera_object)

i = 0
pi=math.pi

def point_at(obj, target, roll=0):
    """
    Rotate obj to look at target

    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians. 

    Based on: https://blender.stackexchange.com/a/5220/12947 (ideasman42)      
    """
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)
    loc = obj.location
    # direction points from the object to the target
    direction = target - loc

    quat = direction.to_track_quat('-Z', 'Y')
    
    # /usr/share/blender/scripts/addons/add_advanced_objects_menu/arrange_on_curve.py
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, 'Z')

    # remember the current location, since assigning to obj.matrix_world changes it
    loc = loc.to_tuple()
    #obj.matrix_world = quat * rollMatrix
    # in blender 2.8 and above @ is used to multiply matrices
    # using * still works but results in unexpected behaviour!
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc
    
for obj_ in bpy.data.objects:
    if obj_.type=="CAMERA":

        ratio=i/cams_num
        angle=(2*pi)*ratio
        Rx=0.3
        Ry=1
        z=1.80
        scl=0.05
        
        #x=-2
        #y=-2+4*ratioo
        #obj_.rotation_euler.z=pi/2
        
        x=math.cos(angle)*Rx-3
        y=math.sin(angle)*Ry

        obj_.location=(x,y,z)
        

        
        slide=0.5
        tar_y=-slide+slide*2*math.sin(ratio*pi)
        target=(0,tar_y,0.5)
        point_at(obj_, target)
        obj_.scale=(scl,scl,scl)
        
        marker = bpy.data.scenes[0].timeline_markers.new('F_'+str(i), frame=i)
        marker.camera = obj_
        
        # collection of cameras
        obj_old_coll = obj_.users_collection
        coll.objects.link(obj_)
        for ob in obj_old_coll: #unlink from all  precedent obj collections
            ob.objects.unlink(obj_)
    
        i=i+1
        
