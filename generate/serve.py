import os
import torch
import time
from PIL import Image
import argparse
from fastapi import FastAPI, HTTPException, Body
import uvicorn
import trimesh
from trimesh.exchange.gltf import export_glb
import numpy as np

from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer

app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lite", default=False, action="store_true")
    parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
    parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
    parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
    parser.add_argument("--save_folder", default="outputs/", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--t2i_seed", default=0, type=int)
    parser.add_argument("--t2i_steps", default=25, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--gen_steps", default=12, type=int)
    parser.add_argument("--max_faces_num", default=10000, type=int)
    parser.add_argument("--save_memory", default=False, action="store_true")
    parser.add_argument("--do_texture_mapping", default=False, action="store_true")
    parser.add_argument("--do_render", default=False, action="store_true")
    parser.add_argument("--port", default=8093, type=int)
    return parser.parse_args()

args = get_args()

# Initialize models globally
rembg_model = Removebg()
image_to_views_model = Image2Views(device=args.device, use_lite=args.use_lite)
views_to_mesh_model = Views2Mesh(args.mv23d_cfg_path, args.mv23d_ckt_path, args.device, use_lite=args.use_lite)
text_to_image_model = Text2Image(pretrain=args.text2image_path, device=args.device, save_memory=args.save_memory)
if args.do_render:
    gif_renderer = GifRenderer(device=args.device)

def process_fake_image_to_3d(image_path, output_folder):
    img = Image.open(image_path)

    # Parameters
    radius = 1.0     # Radius of column
    height = 1.0     # Height of the column
    sides = 6        # Number of vertical faces
    scene = trimesh.Scene()

    # Helper: Generate heptagon points
    angles = np.linspace(np.pi * 24 / 180, 2 * np.pi, sides, endpoint=False)
    vertices_bottom = np.stack([np.cos(angles) * radius, np.zeros(sides), np.sin(angles) * radius], axis=1)
    vertices_top = vertices_bottom + np.array([0, height, 0])

    # Create each vertical face as a quad
    for i in range(sides):
        # Next index (wrap around)
        j = (i + 1) % sides

        # 4 corners of the quad (counter-clockwise order for outward normal)
        v0 = vertices_bottom[i]
        v1 = vertices_bottom[j]
        v2 = vertices_top[j]
        v3 = vertices_top[i]
        face_vertices = np.array([v0, v1, v2, v3])

        # Corrected winding order for front-facing face
        face = trimesh.Trimesh(
            vertices=face_vertices,
            faces=[[0, 2, 1], [0, 3, 2]],
            process=False
        )

        # UV coordinates to map entire image
        uv_coords = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        face.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=img)

        # Add to scene
        scene.add_geometry(face)

    # Export combined mesh
    combined = scene.dump(concatenate=True)
    glb_data = export_glb(combined)
    with open(os.path.join(output_folder, "mesh.glb"), "wb") as f:
        f.write(glb_data)

    print("✅ Exported heptagon_column_textured.glb")


def process_image_to_3d(res_rgb_pil, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Stage 2: Remove Background
    # res_rgba_pil = rembg_model(res_rgb_pil)
    print(f"png---------->: {res_rgb_pil}")
    process_fake_image_to_3d(res_rgb_pil, output_folder)

    print(f"🔥 Image to 3D gen stpes: {args.gen_steps}")
    # Stage 3: Image to Views
    # (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
    #     res_rgba_pil,
    #     seed=args.gen_seed,
    #     steps=args.gen_steps
    # )
    # views_grid_pil.save(os.path.join(output_folder, "views.jpg"))

    # # Stage 4: Views to Mesh
    # views_to_mesh_model(
    #     views_grid_pil,
    #     cond_img,
    #     seed=args.gen_seed,
    #     target_face_count=args.max_faces_num,
    #     save_folder=output_folder,
    #     do_texture_mapping=args.do_texture_mapping
    # )

    # Stage 5: Render GIF
    # if args.do_render:
    #     gif_renderer(
    #         os.path.join(output_folder, 'mesh.obj'),
    #         gif_dst_path=os.path.join(output_folder, 'output.gif'),
    #     )

@app.post("/generate_from_text")
async def text_to_3d(prompt: str = Body()):
    output_folder = os.path.join(args.save_folder, "text_to_3d")
    os.makedirs(output_folder, exist_ok=True)

    print(f"🧩Generating 3D model for args: {args.t2i_seed}, {args.t2i_steps}")
    # Stage 1: Text to Image
    start = time.time()
    res_rgb_pil = text_to_image_model(
        prompt,
        seed=args.t2i_seed,
        steps=args.t2i_steps
    )
    res_rgb_pil.save(os.path.join(output_folder, "mesh.png"))
    
    process_image_to_3d(os.path.join(output_folder, "mesh.png"), output_folder)

    with open(f"{output_folder}/prompts.txt", "a") as file:
        file.write(f"{prompt}\n")
    
    print(f"Successfully generated: {output_folder}")
    print(f"Generation time: {time.time() - start}")

    return {"success": True, "path": output_folder}

@app.post("/generate_from_image")
async def image_to_3d(image_path: str):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Image file not found")

    output_folder = os.path.join(args.save_folder, "image_to_3d")
    os.makedirs(output_folder, exist_ok=True)

    # Load Image
    res_rgb_pil = Image.open(image_path)
    process_image_to_3d(res_rgb_pil, output_folder)

    return {"message": "3D model generated successfully from image", "output_folder": output_folder}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)