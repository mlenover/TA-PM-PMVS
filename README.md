# TA-PM-PMVS
An extension to PM-PMVS for textured, anisotropic surfaces

The extrinsic camera matricies are saved in data/poses.npy. For more information on how to generate a new camera matrix, please see the original implementation at https://github.com/za-cheng/PM-PMVS.

Install all dependencies as indicated in the PM-PMVS repository. Install Blender from https://www.blender.org/download/. Open params.py and set the blender install directory variable to where you have installed Blender.

For improved computational speed, each object should be rendered first. Open the associated blend file for EACH of the object(s) you are planning on evaluating (for example, bear_scene_template.blend). Access the scripting tab, and select "Run Script" (the play arrow at the top of the screen). The render should take a few minutes.

You can optionally edit test Scenarios.csv to indicate the shape, anisotropy, blur algorithm and kernel size that are tested.
