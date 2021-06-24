import os
from os.path import join

import torch
import torch.nn.functional as F
import numpy as np

PATCH_COLOR_DEF = torch.rand([1, 798, 3]) # 32x32 posmap for smpl has 798 valid pixels


def getIdxMap_torch(img, offset=False):
    # img has shape [channels, H, W]
    C, H, W = img.shape
    import torch
    idx = torch.stack(torch.where(~torch.isnan(img[0])))
    if offset:
        idx = idx.float() + 0.5
    idx = idx.view(2, H * W).float().contiguous()
    idx = idx.transpose(0, 1)

    idx = idx / (H-1) if not offset else idx / H
    return idx


def load_masks(PROJECT_DIR, posmap_size, body_model='smpl'):
    uv_mask_faceid = np.load(join(PROJECT_DIR, 'assets', 'uv_masks', 'uv_mask{}_with_faceid_{}.npy'.format(posmap_size, body_model))).reshape(posmap_size, posmap_size)
    uv_mask_faceid = torch.from_numpy(uv_mask_faceid).long().cuda()
    
    smpl_faces = np.load(join(PROJECT_DIR, 'assets', 'smpl_faces.npy')) # faces = triangle list of the body mesh
    flist = torch.tensor(smpl_faces.astype(np.int32)).long()
    flist_uv = get_face_per_pixel(uv_mask_faceid, flist).cuda() # Each (valid) pixel on the uv map corresponds to a point on the SMPL body; flist_uv is a list of these triangles

    points_idx_from_posmap = np.load(join(PROJECT_DIR, 'assets', 'uv_masks', 'idx_smpl_posmap{}_uniformverts_retrieval.npy'.format(posmap_size)))
    points_idx_from_posmap = torch.from_numpy(points_idx_from_posmap).cuda()

    uv_coord_map = getIdxMap_torch(torch.rand(3, posmap_size, posmap_size)).cuda()
    uv_coord_map.requires_grad = True

    return flist_uv, points_idx_from_posmap, uv_coord_map


def get_face_per_pixel(mask, flist):
    '''
    :param mask: the uv_mask returned from posmap renderer, where -1 stands for background
                 pixels in the uv map, where other value (int) is the face index that this
                 pixel point corresponds to.
    :param flist: the face list of the body model,
        - smpl, it should be an [13776, 3] array
        - smplx, it should be an [20908,3] array
    :return:
        flist_uv: an [img_size, img_size, 3] array, each pixel is the index of the 3 verts that belong to the triangle
    Note: we set all background (-1) pixels to be 0 to make it easy to parralelize, but later we
        will just mask out these pixels, so it's fine that they are wrong.
    '''
    mask2 = mask.clone()
    mask2[mask == -1] = 0 #remove the -1 in the mask, so that all mask elements can be seen as meaningful faceid
    flist_uv = flist[mask2]
    return flist_uv


def save_latent_vectors(filepath, latent_vec, epoch):

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(filepath),
    )
    

def load_latent_vectors(filepath, lat_vecs):

    full_filename = filepath

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):
        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_model(path, model, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)


def tensor2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()


def customized_export_ply(outfile_name, v, f = None, v_n = None, v_c = None, f_c = None, e = None):
    '''
    Author: Jinlong Yang, jyang@tue.mpg.de

    Exports a point cloud / mesh to a .ply file
    supports vertex normal and color export
    such that the saved file will be correctly displayed in MeshLab

    # v: Vertex position, N_v x 3 float numpy array
    # f: Face, N_f x 3 int numpy array
    # v_n: Vertex normal, N_v x 3 float numpy array
    # v_c: Vertex color, N_v x (3 or 4) uchar numpy array
    # f_n: Face normal, N_f x 3 float numpy array
    # f_c: Face color, N_f x (3 or 4) uchar numpy array
    # e: Edge, N_e x 2 int numpy array
    # mode: ascii or binary ply file. Value is {'ascii', 'binary'}
    '''

    v_n_flag=False
    v_c_flag=False
    f_c_flag=False

    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        if type(v_n) == 'torch.Tensor':
            v_n = v_n.detach().cpu().numpy()
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            # warnings.warn("Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype = np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                # warnings.warn("Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype = np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n'%(N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n'%(N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n'%(N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2], \
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2],\
                    f_c[i,0], f_c[i,1], f_c[i,2], f_c[i,3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n'%(e[i,0], e[i,1]))


def vertex_normal_2_vertex_color(vertex_normal):
    # Normalize vertex normal
    import torch
    if torch.is_tensor(vertex_normal):
        vertex_normal = vertex_normal.detach().cpu().numpy()
    normal_length = ((vertex_normal**2).sum(1))**0.5
    normal_length = normal_length.reshape(-1, 1)
    vertex_normal /= normal_length
    # Convert normal to color:
    color = vertex_normal * 255/2.0 + 128
    return color.astype(np.ubyte)


def draw_correspondence(pcl_1, pcl_2, output_file):
    '''
    Given a pair of (minimal, clothed) point clouds which have same #points,
    draw correspondences between each point pair as a line and export to a .ply
    file for visualization.
    '''
    assert(pcl_1.shape[0] == pcl_2.shape[0])
    N = pcl_2.shape[0]
    v = np.vstack((pcl_1, pcl_2))
    arange = np.arange(0, N)
    arange = arange.reshape(-1,1)
    e = np.hstack((arange, arange+N))
    e = e.astype(np.int32)
    customized_export_ply(output_file, v, e = e)


def save_result_examples(save_dir, model_name, result_name, 
                         points, normals=None, patch_color=None, 
                         texture=None, coarse_pts=None,
                         gt=None, epoch=None):
    # works on single pcl, i.e. [#num_pts, 3], no batch dimension
    from os.path import join
    import numpy as np

    if epoch is None:
        normal_fn = '{}_{}_pred.ply'.format(model_name,result_name)
    else:
        normal_fn = '{}_epoch{}_{}_pred.ply'.format(model_name, str(epoch).zfill(4), result_name)
    normal_fn = join(save_dir, normal_fn)
    points = tensor2numpy(points)

    if normals is not None:
        normals = tensor2numpy(normals)
        color_normal = vertex_normal_2_vertex_color(normals)
        customized_export_ply(normal_fn, v=points, v_n=normals, v_c=color_normal)

    if patch_color is not None:
        patch_color = tensor2numpy(patch_color)
        if patch_color.max() < 1.1:
            patch_color = (patch_color*255.).astype(np.ubyte)
        pcolor_fn = normal_fn.replace('pred.ply', 'pred_patchcolor.ply')
        customized_export_ply(pcolor_fn, v=points, v_c=patch_color)
    
    if texture is not None:
        texture = tensor2numpy(texture)
        if texture.max() < 1.1:
            texture = (texture*255.).astype(np.ubyte)
        texture_fn = normal_fn.replace('pred.ply', 'pred_texture.ply')
        customized_export_ply(texture_fn, v=points, v_c=texture)

    if coarse_pts is not None:
        coarse_pts = tensor2numpy(coarse_pts)
        coarse_fn = normal_fn.replace('pred.ply', 'interm.ply')
        customized_export_ply(coarse_fn, v=coarse_pts)

    if gt is not None: 
        gt = tensor2numpy(gt)
        gt_fn = normal_fn.replace('pred.ply', 'gt.ply')
        customized_export_ply(gt_fn, v=gt)