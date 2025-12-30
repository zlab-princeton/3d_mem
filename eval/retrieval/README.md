# Metrics

## Light Field Distance (LFD)

We modify the implementation of [GET3D](https://github.com/nv-tlabs/GET3D) for pre-computing LFD features and calculating LFD distance between query and training shapes.

### Environment
You can first clone the GET3D repo and base on its setup to install LFD dependencies.

we provide a headless version for [LFD](https://github.com/kacperkan/light-field-distance) computation without display server requirement.
```bash
git clone https://github.com/kacperkan/light-field-distance
cd light-field-distance/lfd/3DAlignment/
```
update the `Main.c` and `Makefile` using our provided `./lfd/Main.c`, `./lfd/Makefile` files.
```bash
cd ../..
bash compile.sh
```

### Pre-compute LFD features
You can use lfd_multiprocessing.py to pre-compute LFD features with a mesh txt file list.
```bash
python lfd_multiprocessing.py --file_list Train-Meshes.txt --n_process 64

python create_q_list.py -i DIR -o OUT.txt -t lfd/mesh

```

### Compute LFD distance
We provide a simpler way for retrieval. First, build the LFD database from pre-computed features. Then, compute LFD distance between query dataset and the training database. E.g.,
```bash
python collecting_lfd_db.py --gen_list Train-LFD.txt  --db_root  /path/to/generated/ --keep_level 2  --out_dir path/to/LFD_db/output --workers 64 --device cpu --model_id_level 0

python lfd_db_retrieval.py \
  --train_db_dir /path/to/train_lfd_db \
  --test_db_dir  /path/to/test_lfd_db \
  --output       lfd_retrieval_results.json \
  --topk         1 \
  --tgt-chunk    256 \
  --batch-size   32

```


## Uni3D

### Pre-compute Uni3D features
We adopt the same pre-compute then retrieval pipeline for Uni3D metric as LFD. Please refer to the [Uni3D official repository](https://github.com/baaivision/Uni3D) for its environments installation and usage. We pad color channels to zero when computing Uni3D features for no-color point clouds. You can sub-sample to 4096 or 10000 points for Uni3D encoding.

After pre-computing Uni3D features for training and test shapes, you can use the following script to compute Uni3D distance:
```bash
python collecting_emb_db.py \
  --db_root /path/to/uni3d_emb \
  --output_dir /path/to/uni3d_db_output \
  --embedding_name uni3d_embedding.npy \
  --output_prefix uni3d_db \
  --model_id_level 2 \
  --keep_level 2

```


# Contact
Please feel free to contact, since the codebase is modified from the cluster version, there might have some bugs. If you have any questions, please open an issue.