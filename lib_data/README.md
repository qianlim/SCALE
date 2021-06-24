This folder contains example scripts that pre-processes and packs the data into the format required by SCALE. 

First, download the [example paired data](https://owncloud.tuebingen.mpg.de/index.php/s/3jJCdxEzD4HXW8y), unzip it under the `data/` folder, such that the data file paths are `<SCALE repo folder>/data/raw/<data obj files>`. The sample data contain:
- A clothed body mesh (single-layered, as in the case of a real scan),
- A fitted, unclothed SMPL body mesh for this clothed body.

The following instructions will walk you through the data processing. The commands are to be executed in the `SCALE` repository folder.

### Pack the data for the SCALE data loader

The example code `lib_data/pack_data_example.py` will take a pair of {clothed body mesh, unclothed body mesh}, and pack them into the `.npz` file as required by SCALE:

```bash
python lib_data/pack_data_example.py
```

Essentially, the code:

- saves the vertex locations of the minimally-clothed SMPL body, which is used for calculating local coordinate systems in SCALE;
- renders a UV positional map from the minimally-clothed body (see below for details), which serves as the input to the SCALE network;
- uniformly samples a specified number of points (together with their normals) on the clothed body mesh, which serves as the ground truth clothed body for SCALE training. 

The generated `.npz` file will be saved at `<SCALE repo folder>/data/packed/example/`. 


### Get UV positional maps given a SMPL body
This section explains the UV positional map rendering step above in more details. The following command won't generate the data needed for training; instead, it runs an example script that demonstrates the rendering of the positional map given the minimally-clothed SMPL template body mesh in `assets/` as input:

```bash
cd lib_data/posmap_generator/
python apps/example.py
```

The script will generate the following outputs under `lib_data/posmap_generator/example_outputs/`:

- `template_mesh_uv_posmap.png`: the UV positional map, i.e. the input to the SCALE model. Each valid pixel (see below) on this map corresponds to a point on the SMPL body; while their 3D location in R^3 may vary w.r.t. body pose, their relative locations on the body surface (2-manifold) are fixed. The 3-dimensional pixel value of a valid pixel is the (x, y, z) coordinate of its position in R^3. For a clearer demonstration, the default resolution is set to 128x128. In the paper we use the resolution 32x32 for a balance between speed and performance.
- `template_mesh_uv_mask.png`: the binary UV mask of the SMPL model. The white region correspond to the valid pixels on the UV positional map.
- `template_mesh_uv_in3D.obj`: a point cloud consisting the points that correspond to the valid pixels from the UV positional map.

### Note: data normalization

When processing multiple data examples into a dataset, make sure the data are normalized. In the SCALE paper, we normalize the data by:
- setting global orientation to 0 (as they are hardly relevant to the pose-dependent clothing deformation);
- aligning the pelvis location of all data frames. Given the fitted SMPL body, the SMPL model is able to regress the body joint locations. Then simply offset all the meshes by the location of their root joint, i.e. the pelvis. If you need a detailed instruction on this, raise a issue on this repository, or refer to the [SMPL tutorials](https://smpl-made-simple.is.tue.mpg.de/), or contact the [SMPL supporting team](smpl@tuebingen.mpg.de).


### Miscellaneous
Part of the functions under the folder `render_posmap` are adapted from the [PIFu repository](https://github.com/shunsukesaito/PIFu/tree/master/lib/renderer) (MIT license).
