import os
from os.path import join, basename, dirname, realpath
import glob

import numpy as np
import open3d as o3d
from tqdm import tqdm


def render_pcl_front_view(vis, cam_params=None, fn=None, img_save_fn=None, pt_size=3):
    mesh = o3d.io.read_point_cloud(fn)  
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    
    opt.point_size = pt_size

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(img_save_fn, True)

    vis.clear_geometries()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--color_opt', default=None, choices=['normal', 'patch'], help='Color the poins by their normals or group them by patches. Leave it the default None will render both colors.')
    parser.add_argument('-n', '--model_name', default='SCALE_demo_00000_simuskirt', help='Name of the model (experiment), the same as the --name flag in the main experiment')
    parser.add_argument('-r', '--img_res', type=int ,default=1024, help='Resolution of rendered image')
    args = parser.parse_args()

    img_res = args.img_res

    # path for saving the rendered images
    SCRIPT_DIR = dirname(realpath(__file__))
    target_root = join(SCRIPT_DIR, '..', 'results', 'rendered_imgs')
    os.makedirs(target_root, exist_ok=True)
    
    # set up camera
    focal_length = 900 * (args.img_res / 1024.) # 900 is a hand-set focal length when the img resolution=1024. 
    x0, y0 = (img_res-1)/2, (img_res-1)/2
    INTRINSIC = np.array([
        [focal_length, 0.,           x0], 
        [0.,           focal_length, y0],
        [0.,           0.,            1]
    ])

    EXTRINSIC = np.load(join(SCRIPT_DIR, 'cam_front_extrinsic.npy'))


    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    cam_intrinsics.intrinsic_matrix = INTRINSIC
    cam_intrinsics.width = img_res
    cam_intrinsics.height = img_res

    cam_params_front = o3d.camera.PinholeCameraParameters()
    cam_params_front.intrinsic = cam_intrinsics
    cam_params_front.extrinsic = EXTRINSIC

    # configs for the rendering
    render_opts = {
        'normal': ('normal_colored', 3.0, '_pred.ply'),
        'patch': ('patch_colored', 4.0, '_pred_patchcolor.ply'),
    }

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_res, height=img_res)

    # render
    for opt in render_opts.keys():
        if (args.color_opt is not None) and (opt != args.color_opt):
            continue

        color_mode, pt_size, ext = render_opts[opt]
        
        render_savedir = join(target_root, args.model_name, color_mode)
        os.makedirs(render_savedir, exist_ok=True)

        ply_folder = join(target_root, '..', 'saved_samples', args.model_name, 'test')
        print('parsing pcl files at {}..'.format(ply_folder))
        flist = sorted(glob.glob(join(ply_folder, '*{}'.format(ext))))

        for fn in tqdm(flist):
            bn = basename(fn)
            img_save_fn = join(render_savedir, bn.replace('{}'.format(ext), '.png'))
            render_pcl_front_view(vis, cam_params_front, fn, img_save_fn, pt_size=pt_size)


if __name__ == '__main__':
    main()