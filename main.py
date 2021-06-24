import os
from os.path import join, basename, dirname, realpath
import sys
import time
from datetime import date, datetime
import math

PROJECT_DIR = dirname(realpath(__file__))
LOGS_PATH = join(PROJECT_DIR, 'checkpoints')
SAMPLES_PATH = join(PROJECT_DIR, 'results', 'saved_samples')
sys.path.append(PROJECT_DIR)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from lib.config_parser import parse_config
from lib.dataset import CloDataSet   
from lib.network import SCALE
from lib.train_eval_funcs import train, test
from lib.utils_io import load_masks, save_model, save_latent_vectors, load_latent_vectors
from lib.utils_model import SampleSquarePoints
from lib.utils_train import adjust_loss_weights

torch.manual_seed(12345)
np.random.seed(12345)

DEVICE = torch.device('cuda')

def main():
    args = parse_config()

    exp_name = args.name
    
    # NOTE: when using your custom data, modify the following path to where the packed data is stored.
    data_root = join(PROJECT_DIR, 'data', 'packed', '{}'.format(args.data_root))

    log_dir = join(PROJECT_DIR,'tb_logs/{}/{}'.format(date.today().strftime('%m%d'), exp_name))
    ckpt_dir = join(LOGS_PATH, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    samples_dir_val = join(SAMPLES_PATH, exp_name, 'val')
    samples_dir_test = join(SAMPLES_PATH, exp_name, 'test')
    os.makedirs(samples_dir_test, exist_ok=True)
    os.makedirs(samples_dir_val, exist_ok=True)

    flist_uv, valid_idx, uv_coord_map = load_masks(PROJECT_DIR, args.img_size, body_model='smpl')
    
    # build_model
    model = SCALE(
                input_nc=3,
                output_nc_unet=args.pix_feat_dim,
                img_size=args.img_size,
                hsize=args.hsize,
                nf=args.nf,
                up_mode=args.up_mode,
                use_dropout=bool(args.use_dropout),
                pos_encoding=bool(args.pos_encoding),
                num_emb_freqs=args.num_emb_freqs,
                posemb_incl_input=bool(args.posemb_incl_input),
                uv_feat_dim=2
                )
    print(model)

    subpixel_sampler = SampleSquarePoints(npoints=args.npoints,
                                          include_end=bool(args.pq_include_end),
                                          min_val=args.pqmin,
                                          max_val=args.pqmax)

    # Below: for historical reason we have a 256D latent code in the network that is shared for all examples of each garment type.
    # Since SCALE is a garment-specific model, this latent code is therefore the same for all examples that the model sees,
    # and can be safely seen as part of the network parameters.
    lat_vecs = torch.nn.Embedding(1, args.latent_size, max_norm=1.0).cuda()
    torch.nn.init.normal_(lat_vecs.weight.data, 0.0, 1e-2 / math.sqrt(args.latent_size))

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr,},
            {"params": lat_vecs.parameters(), "lr": args.lr,},
        ])

    n_epochs = args.epochs
    epoch_now = 0
    '''
    ------------ Load checkpoints in case of test or resume training ------------
    '''
    if args.mode.lower() in ['test', 'resume']:
        checkpoints = sorted([fn for fn in os.listdir(ckpt_dir) if fn.endswith('_model.pt')])
        latest = join(ckpt_dir, checkpoints[-1])
        print('Loading checkpoint {}'.format(basename(latest)))
        ckpt_loaded = torch.load(latest)

        model.load_state_dict(ckpt_loaded['model_state'])

        checkpoints = sorted([fn for fn in os.listdir(ckpt_dir) if fn.endswith('_latent_vecs.pt')])
        checkpoint = join(ckpt_dir, checkpoints[-1])
        load_latent_vectors(checkpoint, lat_vecs)

        if args.mode.lower() == 'resume':
            optimizer.load_state_dict(ckpt_loaded['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(DEVICE)
            epoch_now = ckpt_loaded['epoch'] + 1
            print('Resume training from epoch {}'.format(epoch_now))

        if args.mode.lower() == 'test':
            epoch_idx = ckpt_loaded['epoch']
            model.to(DEVICE)
            print('Test model with checkpoint at epoch {}'.format(epoch_idx))


    '''
    ------------ Training over or resume from saved checkpoints ------------
    '''
    if args.mode.lower() in ['train', 'resume']:
        train_set = CloDataSet(root_dir=data_root, split='train', sample_spacing=args.data_spacing,
                                img_size=args.img_size, scan_npoints=args.scan_npoints)
        val_set = CloDataSet(root_dir=data_root, split='val', sample_spacing=args.data_spacing,
                                img_size=args.img_size, scan_npoints=args.scan_npoints)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        writer = SummaryWriter(log_dir=log_dir)

        print("Total: {} training examples, {} val examples. Training started..".format(len(train_set), len(val_set)))


        model.to(DEVICE)
        start = time.time()
        pbar = range(epoch_now, n_epochs)
        for epoch_idx in pbar:
            wdecay_rgl = adjust_loss_weights(args.w_rgl, epoch_idx, mode='decay', start=args.decay_start, every=args.decay_every)
            wrise_normal = adjust_loss_weights(args.w_normal, epoch_idx,  mode='rise', start=args.rise_start, every=args.rise_every)

            train_stats = train(
                                model, lat_vecs, DEVICE, train_loader, optimizer,
                                flist_uv, valid_idx, uv_coord_map,
                                subpixel_sampler=subpixel_sampler,
                                w_s2m=args.w_s2m, w_m2s=args.w_m2s, w_rgl=wdecay_rgl,
                                w_latent_rgl=1.0, w_normal=wrise_normal,
                                )

            if epoch_idx % 100 == 0 or epoch_idx == n_epochs - 1:
                ckpt_path = join(ckpt_dir, '{}_epoch{}_model.pt'.format(exp_name, str(epoch_idx).zfill(5)))
                save_model(ckpt_path, model, epoch_idx, optimizer=optimizer)
                ckpt_path = join(ckpt_dir, '{}_epoch{}_latent_vecs.pt'.format(exp_name, str(epoch_idx).zfill(5)))
                save_latent_vectors(ckpt_path, lat_vecs, epoch_idx)

            # test on val set every N epochs
            if epoch_idx % args.val_every == 0:
                dur = (time.time() - start) / (60 * (epoch_idx-epoch_now+1))
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print('\n{}, Epoch {}, average {:.2f} min / epoch.'.format(dt_string, epoch_idx, dur))
                print('Weights s2m: {:.1e}, m2s: {:.1e}, normal: {:.1e}, rgl: {:.1e}'.format(args.w_s2m, args.w_m2s, wrise_normal, wdecay_rgl))

                checkpoints = sorted([fn for fn in os.listdir(ckpt_dir) if fn.endswith('_latent_vecs.pt')])
                checkpoint = join(ckpt_dir, checkpoints[-1])
                load_latent_vectors(checkpoint, lat_vecs)
                val_stats = test(model, lat_vecs,
                                DEVICE, val_loader, epoch_idx,
                                samples_dir_val,
                                flist_uv,
                                valid_idx, uv_coord_map,
                                subpixel_sampler=subpixel_sampler,
                                model_name=exp_name,
                                save_all_results=bool(args.save_all_results),
                                )

                tensorboard_tabs = ['model2scan', 'scan2model', 'normal_loss', 'residual_square', 'latent_reg', 'total_loss']
                stats = {'train': train_stats, 'val': val_stats}

                for split in ['train', 'val']:
                    for (tab, stat) in zip(tensorboard_tabs, stats[split]):
                        writer.add_scalar('{}/{}'.format(tab, split), stat, epoch_idx)


        end = time.time()
        t_total = (end - start) / 60
        print("Training finished, duration: {:.2f} minutes. Now eval on test set..\n".format(t_total))
        writer.close()


    '''
    ------------ Test model ------------
    '''
    test_set = CloDataSet(root_dir=data_root, split='test', sample_spacing=args.data_spacing,
                          img_size=args.img_size, scan_npoints=args.scan_npoints)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print('Eval on test data...')
    start = time.time()
    test_m2s, test_s2m, test_lnormal, _, _, _ = test(model, lat_vecs, DEVICE, test_loader, epoch_idx,
                                                    samples_dir_test,
                                                    flist_uv,
                                                    valid_idx, uv_coord_map,
                                                    mode='test',
                                                    subpixel_sampler=subpixel_sampler,
                                                    model_name=exp_name,
                                                    save_all_results=bool(args.save_all_results),
                                                    )

    print('\nDuration: {}'.format(time.time() - start))
    
    testset_result = "Test on test set, {} examples, m2s dist: {:.3e}, s2m dist: {:.3e}, Chamfer total: {:.3e}, normal loss: {:.3e}.\n\n".format(len(test_set), test_m2s, test_s2m, test_m2s+test_s2m, test_lnormal)
    print(testset_result)
        
    with open(join(PROJECT_DIR, 'results', 'results_log.txt'), 'a+') as fn:
        fn.write('{}, n_pq={}, epoch={}\n'.format(args.name, args.npoints, epoch_idx))
        fn.write('\t{}'.format(testset_result))


if __name__ == '__main__':
    main()