# GeneralizableSound-Bem

## Dependencies

- CUDA (match with the version of PyTorch, which is required by Pytorch CUDA Extension in `src/cuda`):
- Pytorch
- tensorboard
- bempp:

```bash
pip install pyopencl
pip install bempp-cl==0.3.2
```

Other dependencies:

```bash
pip install numpy scipy numba meshio matplotlib tqdm commentjson protobuf ipywidgets IPython
pip install plotly scikit-image ninja librosa seaborn
pip install trimesh
```


## Usage

To use the code, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/syunjyan/GeneralizableSound-Bem.git
    cd GeneralizableSound-Bem
    ```

2. Install the required dependencies as mentioned in the Dependencies section.

3. generate the sound data for point sound source: 

    ```bash
    python experiments/neuPAT_fix/generate.py
    ```

4. generate the sound data for phone sound source: 

    ```bash
    python experiments/neuPAT_phone/generate.py
    ```

5. generate the data for sound to train the demo:

    ```bash
    python experiments/demo_xxx/generate.py
    ```

## Directory Structure

`src` contains the source code for the MC-Bem. 
- `src/scene.py` provides the `Scene` class for creating a scene for the MC-Bem to solve the boundary element method (BEM) problem, where we provide some api to create the scene and solve the BEM problem automatically.

`experiment` contains the experiment scenarios and their codes.

`dataset` contains the mesh files, configuration files, and other data files for the scene and the experiment. The structure of the `dataset` directory is as follows:
```
dataset
├── scene_1_name
|   ├── data
│   │   ├── x.pt # bem calculated data
│   │   ├── ...
│   ├── config.json # configuration file for the scene
│   ├── animation_data.npz # used for network to generate animation sound
│   ├── object_1.obj # mesh file used in the scene
│   ├── object_2.obj
│   ├── ...
├── scene_2_name
│   ├── ...
├── ...

```




## How to Create a Scene and Use the MC-Bem （new）

If you need to create a new scene and use the MC-Bem, you can utilize the tools provided in `src/scene.py`, create the `dataset/xxx` directory, add mesh file and modify the `config.json` file in that directory as described below.

### `/src/scene.py`

`scene.py` is a script that contains classes for creating a scene for the MC-Bem to solve the boundary element method (BEM) problem. The `Scene` class provides an API for constructing the scene according to the configuration file, sampling the scene and setting up the BEM problem to
solve. 

The `Scene` class contains the following methods:

#### `Scene.__init__(self, json_path)`

- This method initializes the `Scene` object with the configuration file specified by `json_path`.

#### `Scene.sample(self, max_resize=2, log=False)`

-  This method samples the scene according to the configuration file. The `max_resize` parameter specifies the maximum resize factor for the scene, and the `log` parameter specifies whether to log the sampling process.


Furthermore, we provide some other **helper functions** to help prepare the config and generate the sound data.

#### `initial_config(data_dir, src_sample_num, trg_sample_num, freq_min, freq_max, trg_pos_min, trg_pos_max)`

- This function generates the initial configuration file for the scene, defining the source and target positions, the frequency range, and the default number of samples for the source and target positions.

- The complete structure of the json file will be described below.

#### `config_add_obj(data_dir, obj_name, size, resize = None, rot_axis=None, rot_pos=None, rot_max_deg:float=None, move=None, position=None, vibration=None)`

- This function adds an object to the configuration file, specifying the object name, size, position,  whether and how to resize, rotate, move, or vibrate the object.

#### `generate_sample_scene(data_dir, data_name, src_sample_num = None, trg_sample_num = None , show_scene=False)`

- This function generates the `x-y` data according to the configuration file, and saves the data`*.pt` to the specified directory.

### `/dataset/scene` directory

> The `dataset/scene` directory is not certain, you can create your own directory to store the scene data.


#### `config.json` (necessary)

config.json is a configuration file for the MC-Bem. It contains the following fields:

- obj: a list of objects in the scene, each object may have the following fields:
    - `name`: the file name of the object
    - `size`: float, the size of the object. Scene class will normalize the object size to `size`.
    - `position`(*optional*): list of 3 floats, the position of the object, default is [0, 0, 0]
    - `resize`(*optional*): list of 3 floats, the resize factor of the object
    - `rot_axis`(*optional*): str, i.e. `y`, the rotation axis of the object
    - `rot_pos`(*optional*): list of 3 floats, the rotation axis position of the object
    - `rot_max_deg`(*optional*): float, the maximum rotation degree of the object
    - `move`(*optional*): list of 3 floats, the movement range of the object
    - `vibration`(*optional*): str, i.e. `y` or `-y`, the vibration of the object
- solver: the solver configuration, including the following fields:
    - `src_sample_num`: the number of source samples
    - `trg_sample_num`: the number of target samples
    - `freq_min`: the minimum frequency
    - `freq_max`: the maximum frequency
    - `trg_pos_min`: list of 3 floats, the minimum target position
    - `trg_pos_max`: list of 3 floats, the maximum target position

## How to Generate the Sound

If the scene is set up, you can generate a listenable sound given the listener position. We provide a script `scripts/generate_sound.py` to generate the sound data for training. 

The dataset structure is as follows:

```
dataset
├── scene_1_name
|   ├── animation_data.npz # info of position and fps
|   ├── config.json 
|   ├── your_mesh.obj
|   ├── ...
```

The `animation_data.npz` file contains the following fields:
```python
{
    'fps': 30, # frame per second
    'position': np.array([[r1, t1, p1], [r2, t2, p2], ...]) # the sphere vector of the scene, where r, t, p are the radius, theta, phi of the sphere vector in [0, 1]
}
```

### `scripts/generate_sound.py`

An example of how to generate the sound of Scene "fix" is as follows:

```bash
python scripts/generate_sound.py -d dataset/fix -n 128 -f True
```


If `--figure` is `True`, the figure will save to the `dataset/*/spectrogram` directory.

