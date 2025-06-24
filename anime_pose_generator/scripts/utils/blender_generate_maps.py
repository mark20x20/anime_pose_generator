import bpy
import os
import sys
import math
import mathutils
import argparse

# ===== 引数処理 =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cam_pos', type=str, required=True)
    parser.add_argument('--cam_rot', type=str, required=True)
    parser.add_argument('--frames', type=str, required=True)
    parser.add_argument('--maps', type=str, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

args = parse_args()

fbx_folder = args.input
output_folder = args.output
cam_pos = [float(v) for v in args.cam_pos.split(',')]
cam_rot = [float(v) for v in args.cam_rot.split(',')]
frame_start, frame_end, frame_step = [int(v) for v in args.frames.split(',')]
map_types = args.maps.split(',')

# ===== 出力ディレクトリ作成 =====
output_dirs = {map_type: os.path.join(output_folder, map_type) for map_type in map_types}
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# ===== 不要なデフォルトオブジェクト削除（Cube、Lightなど）=====
for obj in bpy.context.scene.objects:
    if obj.type not in ['CAMERA', 'LIGHT']:
        obj.select_set(True)
    else:
        obj.select_set(False)
bpy.ops.object.delete()

# ===== マテリアル設定 =====
def setup_material(mode):
    mat_name = f"{mode}_Material"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        emission = nodes.new(type='ShaderNodeEmission')
        geometry = nodes.new(type='ShaderNodeNewGeometry')

        if mode == "Depth":
            links.new(geometry.outputs['Position'], emission.inputs['Color'])
        elif mode == "Normal":
            links.new(geometry.outputs['Normal'], emission.inputs['Color'])
        elif mode == "Skeleton":
            emission.inputs['Color'].default_value = (1.0, 0.3, 0.3, 1.0)  # 赤色

        links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat

def apply_material(mat):
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)

# ===== FBX ファイル処理 =====
fbx_files = [f for f in os.listdir(fbx_folder) if f.lower().endswith('.fbx')]

for fbx_file in fbx_files:
    bpy.ops.import_scene.fbx(filepath=os.path.join(fbx_folder, fbx_file))
    base_name = os.path.splitext(fbx_file)[0]

    # カメラ確認・再作成
    cam = bpy.data.objects.get('Camera')
    if not cam:
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.name = 'Camera'
    cam.data.lens = 50
    cam.data.clip_start = 0.1
    cam.data.clip_end = 1000.0

    # ライト確認・再作成
    if "Main_Light" not in bpy.data.objects:
        light_data = bpy.data.lights.new(name="Main_Light", type='SUN')
        light_obj = bpy.data.objects.new(name="Main_Light", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = (10, -10, 10)
        light_data.energy = 5.0

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    center = sum((obj.location for obj in meshes), mathutils.Vector()) / len(meshes) if meshes else mathutils.Vector((0, 0, 0))

    # レンダリング設定
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    for device in prefs.devices:
        device.use = True
    scene.cycles.device = 'GPU'

    for frame in range(frame_start, frame_end + 1, frame_step):
        scene.frame_set(frame)
        cam.location = (center.x + cam_pos[0], center.y + cam_pos[1], center.z + cam_pos[2])
        cam.rotation_euler = (
            math.radians(cam_rot[0]),
            math.radians(cam_rot[1]),
            math.radians(cam_rot[2])
        )

        for map_type in map_types:
            apply_material(setup_material(map_type))
            filename = f"{base_name}_f{frame}_{map_type.lower()}.png"
            scene.render.filepath = os.path.join(output_dirs[map_type], filename)
            bpy.ops.render.render(write_still=True)

    # インポートしたオブジェクトだけ削除（カメラ・ライトは保持）
    for obj in bpy.context.scene.objects:
        if obj.type not in ['CAMERA', 'LIGHT']:
            obj.select_set(True)
        else:
            obj.select_set(False)
    bpy.ops.object.delete()

print("マップ生成が完了しました。")

#blender --background --python C:/AI/anime_pose_generator/scripts/blender_generate_maps.py