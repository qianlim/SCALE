def parse_config(argv=None):
    import configargparse
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.DefaultConfigFileParser
    description = 'articulated bps project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SCALE')

    # general settings                              
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str, default='debug', help='name of a model/experiment. this name will be used for saving checkpoints and will also appear in saved examples')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'resume', 'test'], help='train, resume or test')

    # architecture related
    parser.add_argument('--hsize', type=int, default=256, help='hideen layer size of the ShapeDecoder mlp')
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--use_dropout', type=int, default=0, help='whether use dropout in the UNet')
    parser.add_argument('--up_mode', type=str, default='upconv',  choices=['upconv', 'upsample'], help='the method to upsample in the UNet')
    parser.add_argument('--latent_size', type=int, default=256, help='the size of a latent vector that conditions the unet, leave it untouched (it is there for historical reason)')
    parser.add_argument('--pix_feat_dim', type=int, default=64, help='dim of the pixel-wise latent code output by the UNet')
    parser.add_argument('--pos_encoding', type=int, default=0, help='use Positional Encoding (PE) for uv coords instead of plain concat')
    parser.add_argument('--posemb_incl_input', type=int, default=0, help='if use PE, then include original coords in the positional encoding')
    parser.add_argument('--num_emb_freqs', type=int, default=6, help='if use PE: number of frequencies used in the positional embedding')

    # policy on how to sample each patch (pq coordinates)
    parser.add_argument('--npoints', type=int, default=16, help='a square number: number of points (pq coordinates) to sample in a local pixel patch')
    parser.add_argument('--pq_include_end', type=int, default=1, help='pq value include 1, i.e. [0,1]; else will be [0,1)')
    parser.add_argument('--pqmin', type=float, default=0., help='min val of the pq interval')
    parser.add_argument('--pqmax', type=float, default=1., help='max val of the pq interval')

    # data related
    parser.add_argument('--data_root', type=str, default='00000_simuskirt', help='the path to the "root" of the packed dataset; under this folder there are train/test/val splits.')
    parser.add_argument('--data_spacing', type=int, default=1, help='get every N examples from dataset (set N a large number for fast experimenting)')
    parser.add_argument('--img_size', type=int, default=32, help='size of UV positional map')
    parser.add_argument('--scan_npoints', type=int, default=-1, help='number of points used in the GT point set. By default -1 will use all points (40000);\
                                                                      setting it to another number N will randomly sample N points at each iteration as GT for training.')

    # loss func related
    parser.add_argument('--w_rgl', type=float, default=2e3, help='weight for residual length regularization term')
    parser.add_argument('--w_normal', type=float, default=1.0, help='weight for the normal loss term')
    parser.add_argument('--w_m2s', type=float, default=1e4, help='weight for the Chamfer loss part 1: (m)odel to (s)can, i.e. from the prediction to the GT points')
    parser.add_argument('--w_s2m', type=float, default=1e4, help='weight for the Chamfer loss part 2: (s)can to (m)odel, i.e. from the GT points to the predicted points')

    # training / eval related
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--decay_start', type=int, default=250, help='start to decay the regularization loss term from the X-th epoch')
    parser.add_argument('--rise_start', type=int, default=250, help='start to rise the normal loss term from the X-th epoch')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--decay_every', type=int, default=400, help='decaly the regularization loss weight every X epochs')
    parser.add_argument('--rise_every', type=int, default=400, help='rise the normal loss weight every X epochs')
    parser.add_argument('--val_every', type=int, default=20, help='validate every x epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--save_all_results', type=int, default=0, help='save the entire test set results at inference')

    args, _ = parser.parse_known_args()

    return args