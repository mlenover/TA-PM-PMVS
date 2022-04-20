import trimesh
import numpy as np
from render import render_mesh
from utils import load_brdf
import cv2
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
import math
import subprocess
import imageio
from params import *
import os
import subprocess

def get_cam_location(P, V, height, width):
    w, h = np.meshgrid(range(width), range(height))
    img_coords = np.stack([h, w, np.ones_like(w)], axis=-1).reshape((-1, 3))
    camera_coords = img_coords.dot(np.linalg.inv(P[:3, :3]).T)  # [height*width, 3]
    ray_camera_coords_xyz = (-camera_coords / camera_coords[:,2:]) # [height*width, 3]
    ray_world_coords_xyz = ray_camera_coords_xyz.dot(V[:3,:3]) # [height*width, 3]

    cam_ray_dir = np.array([199, 319, 1])
    cam_ray_dir = cam_ray_dir.dot(np.linalg.inv(P[:3, :3]).T)
    cam_ray_dir = (-cam_ray_dir / cam_ray_dir[2:])
    cam_ray_dir = cam_ray_dir.dot(V[:3,:3])
    cam_ray_dir = cam_ray_dir / np.linalg.norm(cam_ray_dir)
    cam_ray_dir = np.multiply(cam_ray_dir, [1, 1, -1])

    o = np.linalg.inv(V).dot([0, 0, 0, 1])[:3]  # view origin/camera centre
    norm_vec = np.cross(o, o + [0, 0, 1])
    rotated_vec = np.linalg.inv(V).dot(np.append(norm_vec, 1))[:3]
    normal_vec_norm = np.sqrt(sum(norm_vec ** 2))

    proj_of_r_on_n = (np.dot(rotated_vec, norm_vec) / normal_vec_norm ** 2) * norm_vec
    rotated_vec = rotated_vec - proj_of_r_on_n
    c = np.dot(rotated_vec, norm_vec) / np.linalg.norm(rotated_vec) / np.linalg.norm(norm_vec)
    angle = math.degrees(np.arccos(np.clip(c, -1, 1))) - 90.0
    o = np.multiply(o, [1, 1, -1])

    dir_vec = np.multiply(np.linalg.inv(V).dot([0, -1, 0, 1])[:3], [1, 1, -1])
    dir_vec = dir_vec[[0, 2, 1]]
    return o[[0, 2, 1]], cam_ray_dir[[0, 2, 1]], angle


def packInputFile(shape, anisotropy, kernel, algorithm, filename, do_render=False):
    # camera intrinsic matrix
    K = np.array([[0.0000000e+00, -1.1149023e+03, -2.0000000e+02, 0.0000000e+00],
                  [1.1149023e+03, 0.0000000e+00, -3.2000000e+02, 0.0000000e+00],
                  [0.0000000e+00, 0.0000000e+00, -1.0000000e+00, 0.0000000e+00]])

    HEIGHT, WIDTH = 400, 640  # image dimensions

    brdf_str = 'steel'

    # load mesh and normalize
    # mesh = trimesh.load_mesh('data/reading.obj')
    # mesh = trimesh.load_mesh('data/bunny.obj')
    # scale = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()
    # mesh.vertices = mesh.vertices / scale * 10

    Ps = np.load('data/poses.npy')  # camera extrinsic matrices

    camPoses = [get_cam_location(K, P, HEIGHT, WIDTH) for P in Ps]
    camAngles = [[math.degrees(math.atan(camDir[1][2] / math.sqrt(camDir[1][0] ** 2 + camDir[1][1] ** 2))) + 90, 0,
                  math.degrees(math.atan(-camDir[1][0] / camDir[1][1]))] for camDir in camPoses]
    camPoses = [camPose[0] for camPose in camPoses]

    np.save('data/blender', {
        'CamAngles': camAngles,
        'CamPos': camPoses,
    })

    if do_render:
        if shape == "bunny":
            o = subprocess.call([blender_directory+'/blender bunny_scene_template.blend -P bunny_render_blender.py'], shell=True)
        elif shape == "bear":
            o = subprocess.call([blender_directory + '/blender bear_scene_template.blend -P bear_render_blender.py'],
                                shell=True)
        elif shape == "buddha":
            o = subprocess.call([blender_directory + '/blender buddha_scene_template.blend -P buddha_render_blender.py'],
                                shell=True)

    org_imgs = []

    for i in range(len(Ps)):
        im = imageio.imread(shape+'/'+str(anisotropy)+f'/input-{i:02}.png')
        im = np.delete(im, 3, 2)
        org_imgs.append(im)

    org_imgs = np.array(org_imgs)
    org_imgs = np.where(org_imgs<3, np.nan, org_imgs)

    for i, img in enumerate(org_imgs):
        max_val = np.nanmax(img)
        min_val = np.nanmin(img)
        img = (img - min_val)/(max_val - min_val)
        org_imgs[i] = img

    #Smoothing
    org_imgs = np.nan_to_num(org_imgs)

    for i, img in enumerate(org_imgs):
        if algorithm == "average":
            org_imgs[i] = cv2.blur(img, (int(kernel), int(kernel)))
        elif algorithm == "gaussian":
            org_imgs[i] = cv2.GaussianBlur(img, (int(kernel), int(kernel)), 5)
        elif algorithm == "median":
            tmp_img = img * 255
            tmp_img = tmp_img.astype(np.uint8)
            tmp_img = cv2.medianBlur(tmp_img, int(kernel))
            org_imgs[i] = tmp_img / 255

        elif algorithm == "bilateral":
            tmp_img = img * 255
            tmp_img = tmp_img.astype(np.uint8)
            tmp_img = cv2.bilateralFilter(tmp_img, int(kernel), 75, 75)
            org_imgs[i] = tmp_img / 255

        #cv2.imwrite('./' + filename + f'/filtered-input.png', (org_imgs * 255).astype(np.uint8))
        imageio.imwrite(filename + f'/{i:02}-filtered-input.png', org_imgs[i, :, :, 0:3])

    org_imgs = np.where(org_imgs < 0.0001, np.nan, org_imgs)


    # pack the input file, all input files need to follow this format
    np.save('data/input-file', {
        'imgs': org_imgs,  # input images
        'K': K,  # intrinsic images
        'P': Ps,  # camera poses
        'height': HEIGHT,
        'width': WIDTH,
    })