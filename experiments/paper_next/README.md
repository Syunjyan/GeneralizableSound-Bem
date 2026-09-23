# CVMJ relative-transfer data

The new target is the dimensionless magnitude transfer

\[
H(\mathbf{x}, f) =
\frac{|p_{\mathrm{scene}}(\mathbf{x}, f)|}
{|p_{\mathrm{free}}(\mathbf{x}, f)|}.
\]

- BEM uses the analytic free field of the same uniformly vibrating sphere.
  `validate_relative_transfer.py` also compares it against a numerical
  source-only BEM solve.
- FDTD runs the same pressure source and receivers in a homogeneous scene.
- The old global `FDTD2BEM_SCALE = 1/81.22` is not used.
- FFT bin 0 and the exact Nyquist bin are replaced by their nearest resolved
  neighbors (125 Hz and 7875 Hz). This avoids dividing by ill-conditioned
  free-field endpoint spectra and gives BEM/FDTD the same channel semantics.
- Existing magnitude datasets are not overwritten or silently converted.

## Validation

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/paper_next/validate_relative_transfer.py \
  --skip-fdtd --output experiments/paper_next/validation/bem_relative_transfer.json

python experiments/paper_next/validate_relative_transfer.py \
  --skip-bem --gpu 0 \
  --output experiments/paper_next/validation/fdtd_relative_transfer.json
```

The k-Wave CUDA binary requires `libsz.so.2`. In the `gensound` environment it
is provided by `libaec`; make sure `$CONDA_PREFIX/lib` is in
`LD_LIBRARY_PATH`.

## Pro test protocol

`pro_relative_v2.json` freezes unseen source meshes and the random seed. Each invocation
generates one sample and refuses to overwrite an existing sample:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/paper_next/generate_pro_test.py \
  --category general \
  --object-id 100 \
  --template-mesh dataset/fix/my_obstacles/1a153612bcdab3e23cc149415a408229.obj \
  --output ../GeneralizableSound/data/pro_relative_v2
```

Categories:

- `general`: three source meshes whose vertex counts are absent from every
  Pro_1 general-obstacle training mesh;
- `enclosed`: three source meshes whose vertex counts are absent from every
  Pro_1 half-surrounding training mesh;
- `teapot`: FDTD Utah teapots, held-out IDs 7, 12, and 17.

The future relative-transfer training set must exclude all frozen test object
IDs. Old Pro checkpoints predict the old arbitrary-magnitude target and cannot
be evaluated directly against this new test set.

`pro_relative_v1.json` and `data/pro_relative_v1` are retained only as pipeline
smoke assets: their BEM IDs occurred in legacy Pro_1 training and are not valid
final geometry hold-outs.
