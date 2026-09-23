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

`pro_relative_v1.json` freezes object IDs and the random seed. Each invocation
generates one sample and refuses to overwrite an existing sample:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/paper_next/generate_pro_test.py \
  --category general \
  --object-id 16 \
  --template-mesh-dir ../GeneralizableSound/data_big/shapenet/train_mesh \
  --output ../GeneralizableSound/data/pro_relative_v1
```

Categories:

- `general`: BEM general obstacles, held-out IDs 16–18;
- `enclosed`: BEM half-surrounding obstacles, held-out IDs 16–18;
- `teapot`: FDTD Utah teapots, held-out IDs 7, 12, and 17.

The future relative-transfer training set must exclude all frozen test object
IDs. Old Pro checkpoints predict the old arbitrary-magnitude target and cannot
be evaluated directly against this new test set.
