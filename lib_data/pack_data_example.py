import sys
import os
from os.path import join, basename, realpath, dirname

import numpy as np
import trimesh

PROJECT_DIR = dirname(dirname(realpath(__file__)))
sys.path.append(PROJECT_DIR)


def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from lib_data.posmap_generator.lib.renderer.gl.pos_render import PosRender
    
    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos[uv_mask != 0.0]

    uv_pos = uv_pos.reshape(img_size, img_size, 3)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask, face_id


class DataProcessor(object):
    '''
    Example code for processing the paired data of (clothed body mesh, unclothed SMPL body mesh) into the format required by SCALE.
    Try it with the data (to be downloaded) in the data/raw/ folder.
    '''
    def __init__(self, dataset_name='03375_blazerlong',
                 n_sample_scan=40000, posmap_resl=32,
                 uvs=None, faces_uvs=None):
        super().__init__()

        self.uvs = uvs
        self.faces_uvs = faces_uvs

        self.n_sample_scan = n_sample_scan # the number of points to sample on the (i.e. for the GT clothed body point cloud)
        self.posmap_resl = posmap_resl # resolution of the UV positional map

        self.save_root = join(PROJECT_DIR, 'data', 'packed', dataset_name)
        os.makedirs(self.save_root, exist_ok=True)


    def pack_single_file(self, minimal_fn, clothed_fn):
        result = {}
        
        scan = trimesh.load(clothed_fn, process=False)

        scan_pc, faceid = trimesh.sample.sample_surface_even(scan, self.n_sample_scan+100) # sample_even may cause smaller number of points sampled than wanted
        scan_pc = scan_pc[:self.n_sample_scan]
        faceid = faceid[:self.n_sample_scan]
        scan_n = scan.face_normals[faceid]
        result['scan_pc'] = scan_pc
        result['scan_n'] = scan_n
        result['scan_name'] = basename(clothed_fn).replace('.obj', '')

        minimal_mesh = trimesh.load(minimal_fn, process=False)
        posmap, _, _ = render_posmap(minimal_mesh.vertices, minimal_mesh.faces, self.uvs, self.faces_uvs, img_size=self.posmap_resl)
        result['posmap{}'.format(self.posmap_resl)] = posmap
        result['body_verts'] = minimal_mesh.vertices

        save_fn = join(self.save_root, basename(clothed_fn).replace('_clothed.obj', '_packed.npz'))
        np.savez(save_fn, **result)

        return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='example', help='name of the dataset to be created')
    parser.add_argument('--n_sample_scan', type=int, default=40000, help='number of the poinst to sample from the GT clothed body mesh surface')
    parser.add_argument('--posmap_resl', type=int, default=32, help='resolution of the UV positional map to be rendered')
    args = parser.parse_args()

    from lib_data.posmap_generator.lib.renderer.mesh import load_obj_mesh

    uv_template_fn = join(PROJECT_DIR, 'assets', 'template_mesh_uv.obj')
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)

    minimal_fn = join(PROJECT_DIR, 'data', 'raw', 'example_body_minimal.obj')
    clothed_fn = join(PROJECT_DIR, 'data', 'raw', 'example_body_clothed.obj')

    data = DataProcessor(dataset_name=args.dataset_name, uvs=uvs, faces_uvs=faces_uvs, n_sample_scan=args.n_sample_scan, posmap_resl=args.posmap_resl)
    data.pack_single_file(minimal_fn, clothed_fn)

    