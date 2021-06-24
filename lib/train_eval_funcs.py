import torch
from tqdm import tqdm

from lib.utils_io import PATCH_COLOR_DEF, save_result_examples
from lib.losses import normal_loss, chamfer_loss_separate
from lib.utils_model import gen_transf_mtx_full_uv


def train(
        model, lat_vecs, device, train_loader, optimizer,
        flist_uv, valid_idx, uv_coord_map,
        subpixel_sampler=None,
        w_s2m=1e4, w_m2s=1e4, w_normal=10, 
        w_rgl=1e2, w_latent_rgl = 1.0        
        ):

    n_train_samples = len(train_loader.dataset)

    train_s2m, train_m2s, train_lnormal, train_rgl, train_latent_rgl, train_total = 0, 0, 0, 0, 0, 0

    model.train()
    for _, data in enumerate(train_loader):

        # -------------------------------------------------------
        # ------------ load batch data and reshaping ------------

        [inp_posmap, target_pc_n, target_pc, target_names, body_verts, index] = data
        gpu_data = [inp_posmap, target_pc_n, target_pc, body_verts, index]
        [inp_posmap, target_pc_n, target_pc, body_verts, index] = list(map(lambda x: x.to(device), gpu_data))
        bs, _, H, W = inp_posmap.size()

        optimizer.zero_grad()

        transf_mtx_map = gen_transf_mtx_full_uv(body_verts, flist_uv)
        lat_vec_batch = lat_vecs(torch.tensor(0).cuda()).expand(bs, -1)

        uv_coord_map_batch = uv_coord_map.expand(bs, -1, -1).contiguous()
        pq_samples = subpixel_sampler.sample_regular_points() # pq coord grid for one patch
        pq_repeated = pq_samples.expand(bs, H * W, -1, -1) # repeat the same pq parameterization for all patches

        N_subsample = pq_samples.shape[1]

        # The same body point is shared by all sampled pq points within each patch
        bp_locations = inp_posmap.expand(N_subsample, -1, -1,-1,-1).permute([1, 2, 3, 4, 0]) #[bs, C, H, W, N_sample],
        transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute([1, 2, 3, 0, 4, 5])  # [bs, H, W, N_subsample, 3, 3]

        # --------------------------------------------------------------------
        # ------------ model pass an coordinate transformation ---------------

        # Core: predict the clothing residual (displacement) from the body, and their normals
        pred_res, pred_normals = model(inp_posmap, clo_code=lat_vec_batch,
                                       uv_loc=uv_coord_map_batch,
                                       pq_coords=pq_repeated)

        # local coords --> global coords
        pred_res = pred_res.permute([0,2,3,4,1]).unsqueeze(-1)
        pred_normals = pred_normals.permute([0,2,3,4,1]).unsqueeze(-1)

        pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
        pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
        pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
        
        # residual to abosolute locations in space
        full_pred = pred_res.permute([0,4,1,2,3]).contiguous() + bp_locations

        # take the selected points and reshape to [Npoints, 3]
        full_pred = full_pred.permute([0,2,3,4,1]).reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]
        pred_normals = pred_normals.reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]

        # reshaping the points that are grouped into patches into a big point set
        full_pred = full_pred.reshape(bs, -1, 3).contiguous()
        pred_normals = pred_normals.reshape(bs, -1, 3).contiguous()

        # --------------------------------
        # ------------ losses ------------

        # Chamfer dist from the (s)can to (m)odel: from the GT points to its closest ponit in the predicted point set
        m2s, s2m, idx_closest_gt, _ = chamfer_loss_separate(full_pred, target_pc) #idx1: [#pred points]
        s2m = torch.mean(s2m)

        # normal loss
        lnormal, closest_target_normals = normal_loss(pred_normals, target_pc_n, idx_closest_gt)
        
        # dist from the predicted points to their respective closest point on the GT, projected by
        # the normal of these GT points, to appxoimate the point-to-surface distance
        nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
        target_points_chosen = torch.gather(target_pc, dim=1, index=nearest_idx)
        pc_diff = target_points_chosen - full_pred # vectors from prediction to its closest point in gt pcl
        m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
        m2s = torch.mean(m2s**2) # the length (squared) is the approx. pred point to scan surface dist.

        rgl_len = torch.mean(pred_res ** 2)
        rgl_latent = torch.sum(torch.norm(lat_vec_batch, dim=1))

        loss = s2m*w_s2m + m2s*w_m2s  + rgl_len*w_rgl + lnormal* w_normal + rgl_latent*w_latent_rgl

        loss.backward()
        optimizer.step()

        # ------------------------------------------
        # ------------ accumulate stats ------------

        train_m2s += m2s * bs
        train_s2m += s2m * bs
        train_lnormal += lnormal * bs
        train_rgl += rgl_len * bs
        train_latent_rgl += rgl_latent * bs

        train_total += loss * bs

    train_s2m /= n_train_samples
    train_m2s /= n_train_samples
    train_lnormal /= n_train_samples
    train_rgl /= n_train_samples
    train_latent_rgl /= n_train_samples
    train_total /= n_train_samples

    return train_m2s, train_s2m, train_lnormal, train_rgl, train_latent_rgl, train_total


def test(model, lat_vecs, device, test_loader, epoch_idx, samples_dir, 
         flist_uv, valid_idx, uv_coord_map,
         mode='val', subpixel_sampler=None, 
         model_name=None, save_all_results=False):

    model.eval()

    lat_vecs = lat_vecs(torch.tensor(0).cuda()) # it's here for historical reason, can safely treat it as part of the network weights

    n_test_samples = len(test_loader.dataset)

    test_s2m, test_m2s, test_lnormal, test_rgl, test_latent_rgl = 0, 0, 0, 0, 0

    with torch.no_grad():
        for data in tqdm(test_loader):

            # -------------------------------------------------------
            # ------------ load batch data and reshaping ------------

            [inp_posmap, target_pc_n, target_pc, target_names, body_verts, index] = data
            gpu_data = [inp_posmap, target_pc_n, target_pc, body_verts, index]
            [inp_posmap, target_pc_n, target_pc, body_verts, index] = list(map(lambda x: x.to(device, non_blocking=True), gpu_data))
            
            bs, C, H, W = inp_posmap.size()
            
            lat_vec_batch = lat_vecs.expand(bs, -1)
            transf_mtx_map = gen_transf_mtx_full_uv(body_verts, flist_uv)
            uv_coord_map_batch = uv_coord_map.expand(bs, -1, -1).contiguous()

            pq_samples = subpixel_sampler.sample_regular_points()
            pq_repeated = pq_samples.expand(bs, H * W, -1, -1)  # [B, H*W, samples_per_pix, 2]

            N_subsample = pq_samples.shape[1]

            # The same body point is shared by all sampled pq points within each patch
            bp_locations = inp_posmap.expand(N_subsample, -1, -1,-1,-1).permute([1, 2, 3, 4, 0]) # [B, C, H, W, N_sample]
            transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute([1, 2, 3, 0, 4, 5])  # [B, H, W, N_subsample, 3, 3]

            # --------------------------------------------------------------------
            # ------------ model pass an coordinate transformation ---------------

            # Core: predict the clothing residual (displacement) from the body, and their normals
            pred_res, pred_normals = model(inp_posmap, clo_code=lat_vec_batch,
                                           uv_loc=uv_coord_map_batch,
                                           pq_coords=pq_repeated)

            # local coords --> global coords
            pred_res = pred_res.permute([0,2,3,4,1]).unsqueeze(-1)
            pred_normals = pred_normals.permute([0, 2, 3, 4, 1]).unsqueeze(-1)

            pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
            pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
            pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
            
            # residual to abosolute locations in space
            full_pred = pred_res.permute([0,4,1,2,3]).contiguous() + bp_locations
            
            # take the selected points and reshape to [N_valid_points, 3]
            full_pred = full_pred.permute([0,2,3,4,1]).reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]
            pred_normals = pred_normals.reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]

            full_pred = full_pred.reshape(bs, -1, 3).contiguous()
            pred_normals = pred_normals.reshape(bs, -1, 3).contiguous()
            
            # --------------------------------
            # ------------ losses ------------

            _, s2m, idx_closest_gt, _ = chamfer_loss_separate(full_pred, target_pc) #idx1: [#pred points]
            s2m = s2m.mean(1)
            lnormal, closest_target_normals = normal_loss(pred_normals, target_pc_n, idx_closest_gt, phase='test')
            nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
            target_points_chosen = torch.gather(target_pc, dim=1, index=nearest_idx)
            pc_diff = target_points_chosen - full_pred # vectors from prediction to its closest point in gt pcl
            m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
            m2s = torch.mean(m2s**2, 1) 

            rgl_len = torch.mean((pred_res ** 2).reshape(bs, -1),1)
            rgl_latent = torch.sum(torch.norm(lat_vec_batch, dim=1))

            # ------------------------------------------
            # ------------ accumulate stats ------------

            test_m2s += torch.sum(m2s)
            test_s2m += torch.sum(s2m)
            test_lnormal += torch.sum(lnormal)
            test_rgl += torch.sum(rgl_len)
            test_latent_rgl += rgl_latent

            patch_colors = PATCH_COLOR_DEF.expand(N_subsample, -1, -1).transpose(0,1).reshape(-1, 3)

            if mode == 'test':
                save_spacing = 1 if save_all_results else 10
                for i in range(full_pred.shape[0])[::save_spacing]:
                    save_result_examples(samples_dir, model_name, target_names[i],
                                         points=full_pred[i], normals=pred_normals[i],
                                         patch_color=patch_colors)
    
    test_m2s /= n_test_samples
    test_s2m /= n_test_samples
    test_lnormal /= n_test_samples
    test_rgl /= n_test_samples
    test_latent_rgl /= n_test_samples

    test_s2m, test_m2s, test_lnormal, test_rgl, test_latent_rgl = list(map(lambda x: x.detach().cpu().numpy(), [test_s2m, test_m2s, test_lnormal, test_rgl, test_latent_rgl]))
    test_total_loss = test_s2m + test_m2s + test_lnormal + test_rgl + test_latent_rgl

    if mode != 'test':
        if epoch_idx == 0 or epoch_idx % 10 == 0:
            # only save the first example per batch for quick inspection of validation set results
            save_result_examples(samples_dir, model_name, target_names[0],
                                 points=full_pred[0], normals=pred_normals[0],
                                 patch_color=None, epoch=epoch_idx)

    print("model2scan dist: {:.3e}, scan2model dist: {:.3e}, normal loss: {:.3e}"
          " rgl term: {:.3e}, latent rgl term:{:.3e},".format(test_m2s, test_s2m, test_lnormal,
                                                              test_rgl, test_latent_rgl))

    return test_m2s, test_s2m, test_lnormal, test_rgl, test_latent_rgl, test_total_loss
