# GeneralizableSound-Bem

## Dependencies

- CUDA (match with the version of PyTorch, which is required by Pytorch CUDA Extension in `src/cuda`):
- Pytorch
- tensorboard
- tiny-cuda-nn

```bash
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install
```

- Bempp:

```bash
pip install plotly pandas pyopencl[pocl] gmsh
# for windows, install pyopencl from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl
git clone https://github.com/bempp/bempp-cl
cd bempp-cl
python setup.py install
```

Other dependencies:

```bash
pip install numpy scipy numba meshio matplotlib tqdm commentjson protobuf ipywidgets IPython
pip install plotly scikit-image ninja librosa
```
