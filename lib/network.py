import torch
import torch.nn as nn

from lib.utils_model import PositionalEncoding, normalize_uv
from lib.modules import UnetCond5DS, ShapeDecoder

class SCALE(nn.Module):
    def __init__(
                self, 
                input_nc=3, # num channels of the unet input
                output_nc_unet=64, # num channels output by the unet
                cond_dim=256, 
                nf=64, # num filters for the unet
                img_size=32, # size of UV positional map
                hsize=256, # hidden layer size of the ShapeDecoder MLP
                up_mode='upconv', 
                use_dropout=False,
                pos_encoding=False, # whether use Positional Encoding
                num_emb_freqs=8, # number of sinusoida frequences if positional encoding is used
                posemb_incl_input=False, # wheter include the original coordinate if using Positional Encoding
                uv_feat_dim=2, # input dimension of the uv coordinates
                pq_feat_dim = 2 # input dimension of the pq coordinates
                ):

        super().__init__()
        self.cond_dim = cond_dim
        self.output_nc_unet = output_nc_unet
        self.pos_encoding = pos_encoding
        self.num_emb_freqs = num_emb_freqs
        self.img_size = img_size
        
        if self.pos_encoding:
            self.embedder = PositionalEncoding(num_freqs=num_emb_freqs,
                                               input_dims=uv_feat_dim,
                                               include_input=posemb_incl_input)
            self.embedder.create_embedding_fn()
            uv_feat_dim = self.embedder.out_dim

        # U-net: for extracting pixel-aligned pose features from UV positional maps
        self.unet = UnetCond5DS(input_nc, output_nc_unet, nf, cond_dim=cond_dim, up_mode=up_mode,
                                use_dropout=use_dropout)
        
        # core: maps the contatenated features+uv,pq coords to outputs
        self.map2Dto3D = ShapeDecoder(in_size=uv_feat_dim + pq_feat_dim + output_nc_unet,
                                      hsize=hsize, actv_fn='softplus')

    def forward(self, x, clo_code, uv_loc, pq_coords):
        '''
        :param x: input posmap, [batch, 3, 256, 256]
        :param clo_code:  clothing encoding for conditioning [256,], it's here for historical reasons but plays no actual role (as the model is outfit-specific)
        :param uv_loc: uv coordinate between 0,1 for each pixel, [B, N_pix, N_subsample, 2]. at each [B, N_pix], the N_subsample rows are the same, 
                       i.e. all subpixel share the same discrete (u,v) value.
        :param pq_coords: (p,q) coordinates in subpixel space, range [0,1), shape [B, N_pix, N_subsample, 2]
        :returns: 
            residuals and normals of the points that are grouped into patches, both in shape  [B, 3, H, W, N_subsample],
            where N_subsample is the number of points sampled per patch.
        '''
        pix_feature = self.unet(x, clo_code)
        B, C = pix_feature.size()[:2]
        H, W = self.img_size, self.img_size
        N_subsample = pq_coords.shape[2]

        uv_feat_dim = uv_loc.size()[-1]
        pq_coords = pq_coords.reshape(B, -1, 2).transpose(1, 2)  # [B, 2, Num of all pq subpixels]
        uv_loc = uv_loc.expand(N_subsample, -1, -1, -1).permute([1, 2, 0, 3])

        # uv and pix feature are shared for all points within each patch
        pix_feature = pix_feature.view(B, C, -1).expand(N_subsample, -1,-1,-1).permute([1,2,3,0]) # [B, C, N_pix, N_sample_perpix]
        pix_feature = pix_feature.reshape(B, C, -1)

        if self.pos_encoding:
            uv_loc = normalize_uv(uv_loc).view(-1,uv_feat_dim)
            uv_loc = self.embedder.embed(uv_loc).view(B, -1,self.embedder.out_dim).transpose(1,2)
        else:
            uv_loc = uv_loc.reshape(B, -1, uv_feat_dim).transpose(1, 2)  # [B, N_pix, N_subsample, 2] --> [B, 2, Num of all pq subpixels]

        # Core of this func: concatenated inputs --network--> outputs
        residuals, normals = self.map2Dto3D(torch.cat([pix_feature, uv_loc, pq_coords], 1))  # [B, 3, Num of all pq subpixels]

        # shape the output to the shape of
        # [batch, height of the positional map, width of positional map, #sampled points per pixel on the positional map (corresponds to a patch)]
        residuals = residuals.view(B, 3, H, W, N_subsample)
        normals = normals.view(B, 3, H, W, N_subsample)

        return residuals, normals