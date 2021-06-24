This folder contains data examples that illustrate the data format required by SCALE in the `packed` folder, and the "raw" paired data of {clothed posed body mesh, unclothed posed body mesh} in the `raw` folder, for illustrating how to pack the data. Follow the instructions in the repository's main `README` and `lib_data/README` to download the data.

Note: the data provided for downloading are only for demonstration purposes, therefore:

- The data in the `packed` and `raw` folders are not in correspondence.
- There is only a small number of training data frames offered in the `packed` folder; training solely on them will not yield a satisfactory model. Usually training SCALE requires hundreds to thousand of frames to generalize well on unseen poses. For more data, you can download from the [CAPE Dataset website](https://cape.is.tue.mpg.de/dataset) or use your own data, as long as they are processed and packed in the same format as our provided data.

### "packed" folder
Each example is stored as a separate `.npz` file that contains the following fields:
- `posmap32`: UV positional map of the minimally-clothed, posed body;
- `body_verts`: vertices of the minimally-clothed, posed body;
- `scan_pc`: a point set sampled from the corresponding clothed, posed body;
- `scan_n`: the points' normals of the point set above.
- `scan_name`: name of the frame in the format of `<outfit_type>_<sequence_name>.<frame_id>`;

### "raw" folder

This folder contains an example pair of  {clothed posed body mesh, unclothed posed body mesh}. Please refer to `lib_data/README` folder and try it out with the data here.

### License

The data provided in this folder, including the minimally-clothed body shapes, body poses and the shape of the clothed body (in the format of meshes and sampled point clouds), are subject to the [license of the CAPE dataset](https://cape.is.tue.mpg.de/license). The UV map design is from the [SMPL](https://smpl.is.tue.mpg.de/) body model and is subject to the license of SMPL.