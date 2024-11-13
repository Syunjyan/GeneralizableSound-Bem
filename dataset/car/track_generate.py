import bpy
import math
import pymeshlab


def process_and_export(rotation_angle, iteration):

    bpy.ops.object.select_all(action="DESELECT")
    plane = bpy.data.objects["Plane"]
    plane.select_set(True)
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.duplicate(linked=False)
    plane_copy = bpy.context.active_object
    plane_copy.name = "Plane_copy"

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.spin(
        steps=12,
        angle=math.radians(-rotation_angle),
        axis=(0, 0, 1),
        center=bpy.context.scene.cursor.location,
    )
    bpy.ops.object.mode_set(mode="OBJECT")

    line = bpy.data.objects["Line"]
    for modifier in line.modifiers:
        if modifier.type == "BOOLEAN":
            modifier.object = plane_copy

    bpy.ops.object.select_all(action="DESELECT")
    track = bpy.data.objects["Track"]
    track.select_set(True)
    filepath = f"{data_dir}/{iteration}.obj"
    bpy.ops.export_scene.obj(
        filepath=filepath,
        use_selection=True,
        axis_forward="-X",
        axis_up="Z",
        use_materials=False,
        use_normals=False,
        use_uvs=False,
    )

    bpy.data.objects.remove(plane_copy, do_unlink=True)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filepath)
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(1.2))
    ms.save_current_mesh(filepath.replace(".obj", "_remesh.obj"))


data_dir = "/home/jxt/NeuralAT/dataset/NeuPAT_new/car/tracks/"

for i in range(180):
    process_and_export(45 + i, i)
