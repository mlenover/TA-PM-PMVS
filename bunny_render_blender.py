import numpy as np
import bpy
from pathlib import Path
import math

# path = some_object.filepath # as declared by props
path = "//data/blender.npy"  # a blender relative path

f = Path(bpy.path.abspath(path))  # make a path object of abs path

if f.exists():
    camData = np.load(f, allow_pickle=True).item()

scene = bpy.data.scenes["Scene"]
layer = bpy.context.view_layer

bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 640  # perhaps set resolution in code
bpy.context.scene.render.resolution_y = 400

for anisotropy in [0, 0.2, 0.4, 0.6]:
    i = 0
    print(Path(bpy.path.abspath("//bunny/" + str(anisotropy) + f"/input-{i:02}.png")))

    print(camData["CamPos"][0])

    cam = bpy.data.objects['Camera']
    light_object = bpy.data.objects["Light"]
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.use_compositing = True

    for i in range(len(camData["CamPos"])):
        bpy.context.scene.render.resolution_percentage = 100
        cam.location.x = camData["CamPos"][i][0]
        cam.location.y = camData["CamPos"][i][1]
        cam.location.z = camData["CamPos"][i][2]

        cam.rotation_euler[0] = math.radians(camData["CamAngles"][i][0])
        cam.rotation_euler[1] = math.radians(camData["CamAngles"][i][1])
        cam.rotation_euler[2] = math.radians(camData["CamAngles"][i][2])

        light_object.location.x = camData["CamPos"][i][0]
        light_object.location.y = camData["CamPos"][i][1]
        light_object.location.z = camData["CamPos"][i][2]

        bpy.context.scene.node_tree.nodes.active
        bpy.data.materials["Default OBJ.001"].node_tree.nodes["Anisotropic BSDF"].inputs[2].default_value = anisotropy
        bpy.context.scene.render.filepath = str(Path(bpy.path.abspath("//bunny/" + str(anisotropy) + f"/input-{i:02}.png")))
        # scene.node_tree.nodes["File Output"].base_path = str(Path(bpy.path.abspath(f"//results/depth-{i:02}")))
        bpy.ops.render.render(write_still=True)
        bpy.ops.render.render()

        z = bpy.data.images['Viewer Node']
        dmap = np.array(z.pixels[:], dtype=np.float32)
        dmap = np.reshape(dmap, [400, 640, 4])
        dmap = dmap[:, :, 0]
        dmap = np.where(dmap > 1.e+09, np.inf, dmap)
        dmap = np.flipud(dmap)
        np.save(str(Path(bpy.path.abspath("//bunny/" + str(anisotropy) + f"/depth-{i:02}"))), dmap)

bpy.ops.wm.quit_blender()