from .modalobj.model import StaticObj, get_mesh_center, normalize_vertices
import json
from scipy.spatial.transform import Rotation as R
import torch
from .utils import Visualizer
import os
from src.bem.solver import BEM_Solver
import numpy as np
from glob import glob
import meshio
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def rotate_vertices(vs, rot_axis, rot_degree):
    vs = R.from_euler(rot_axis, rot_degree, degrees=True).apply(vs.cpu().numpy())
    return torch.from_numpy(vs).cuda()


class ObjList:
    def __init__(self, obj_json, data_dir):
        self.vertices_list = []
        self.triangles_list = []
        self.neumann_list = []
        obj_dir = os.path.join(data_dir, obj_json["mesh_dir"])
        print("Loading obj files")
        for obj_path in tqdm(glob(os.path.join(obj_dir, "*_remesh.obj"))):
            obj = StaticObj(obj_path, obj_json["size"])
            self.vertices_list.append(
                torch.tensor(obj.vertices).cuda().to(torch.float32)
            )
            self.triangles_list.append(
                torch.tensor(obj.triangles).cuda().to(torch.int32)
            )
            neumann = torch.zeros(
                len(obj.vertices), dtype=torch.complex64, device="cuda"
            )
            self.neumann_list.append(neumann)
        self.obj_num = len(self.vertices_list)
        self.rot_vec = None
        self.move_vec = None

    def sample(self, rnd):
        idx = int(rnd * self.obj_num)
        self.triangles = self.triangles_list[idx]
        self.vertices = self.vertices_list[idx]
        self.neumann = self.neumann_list[idx]

    def reset(self):
        pass

    def resize(self, factor):
        pass

    def rotation(self, factor):
        pass

    def move(self, factor):
        pass


class ObjAnim:
    def __init__(self, obj_json, data_dir):
        obj = StaticObj(os.path.join(data_dir, obj_json["mesh"]), obj_json["size"])
        self.vertices_base = obj.vertices
        self.triangles = torch.tensor(obj.triangles).cuda().to(torch.int32)
        self.trajectory_points = meshio.read(
            os.path.join(data_dir, obj_json["trajectory"])
        ).points
        self.trajectory_points = normalize_vertices(
            self.trajectory_points, obj_json["trajectory_size"]
        )
        self.trajectory_length = len(self.trajectory_points) - 1
        self.vibration = None if "vibration" not in obj_json else obj_json["vibration"]

        self.neumann = torch.zeros(
            len(self.vertices_base), dtype=torch.complex64, device="cuda"
        )
        if self.vibration is not None:
            self.neg = False
            if "-" in self.vibration:
                self.neg = True
                self.vibration = self.vibration.replace("-", "")
            idx = 0 if "x" in self.vibration else 1 if "y" in self.vibration else 2
            if self.neg:
                self.neumann[self.vertices_base[:, idx] < 0] = 1
            else:
                self.neumann[self.vertices_base[:, idx] > 0] = 1
        self.offset = None if "offset" not in obj_json else obj_json["offset"]

    def sample(self, rnd):
        idx = int(rnd * self.trajectory_length)
        x0 = self.trajectory_points[idx]
        x1 = self.trajectory_points[idx + 1]
        v = x1 - x0
        center = (x0 + x1) / 2 + self.offset
        x_axis = np.array([1, 0, 0])
        rotation, _ = R.align_vectors([v], [x_axis])
        self.vertices = rotation.apply(self.vertices_base)
        self.vertices = self.vertices - get_mesh_center(self.vertices) + center
        self.vertices = torch.tensor(self.vertices).cuda().to(torch.float32)

    def reset(self):
        pass

    def resize(self, factor):
        pass

    def rotation(self, factor):
        pass

    def move(self, factor):
        pass


class Obj:
    def __init__(self, obj_json, data_dir):
        obj = StaticObj(os.path.join(data_dir, obj_json["mesh"]), obj_json["size"])
        self.name = obj_json["mesh"]
        self.vertices_base = torch.tensor(obj.vertices).cuda().to(torch.float32)
        self.triangles = torch.tensor(obj.triangles).cuda().to(torch.int32)
        self.resize_base = (
            torch.zeros(3).cuda()
            if "resize" not in obj_json
            else torch.tensor(obj_json["resize"]).cuda()
        )
        self.rot_axis = None if "rot_axis" not in obj_json else obj_json["rot_axis"]
        self.rot_pos = (
            None
            if "rot_pos" not in obj_json
            else torch.tensor(obj_json["rot_pos"]).cuda()
        )
        self.rot_max_deg = (
            None if "rot_max_deg" not in obj_json else obj_json["rot_max_deg"]
        )
        self.move_vec = (
            None if "move" not in obj_json else torch.tensor(obj_json["move"]).cuda()
        )
        self.position = (
            None
            if "position" not in obj_json
            else torch.tensor(obj_json["position"]).cuda()
        )
        self.vibration = None if "vibration" not in obj_json else obj_json["vibration"]

        self.neumann = torch.zeros(
            len(self.vertices_base), dtype=torch.complex64, device="cuda"
        )
        if self.vibration is not None:
            self.neg = False
            if "-" in self.vibration:
                self.neg = True
                self.vibration = self.vibration.replace("-", "")
            idx = 0 if "x" in self.vibration else 1 if "y" in self.vibration else 2
            if self.neg:
                self.neumann[self.vertices_base[:, idx] < 0] = 1
            else:
                self.neumann[self.vertices_base[:, idx] > 0] = 1

    def reset(self):
        self.vertices = self.vertices_base.clone()

    def resize(self, factor):
        self.resize_vec = self.resize_base * factor
        self.resize_vec[self.resize_vec == 0] = 1
        self.vertices *= self.resize_vec

    def rotation(self, factor):
        if self.rot_axis is not None:
            self.vertices = (
                rotate_vertices(
                    self.vertices - self.rot_pos * self.resize_vec,
                    self.rot_axis,
                    factor * self.rot_max_deg,
                )
                + self.rot_pos * self.resize_vec
            )

    def move(self, factor):
        if self.move_vec is not None:
            self.vertices = self.vertices + self.move_vec * factor
    
    def shift(self):
        if self.position is not None:
            self.vertices += self.position
            
    def move_the_object(self, move_vector: torch.Tensor,
                        rotate_vector: torch.Tensor, 
                        scale_vector: torch.Tensor,
                        ):
        """
        重写的一个较简单的move函数，直接移动/旋转物体到指定位置。
        具体来说，我们首先缩放物体，然后以 (0,0,0) 为中心旋转物体，然后移动物体到指定位置。
        :param move_vector: torch.Tensor, 移动向量, [x, y, z]
        :param rotate_vector: torch.Tensor, 旋转向量, [x, y, z]
        :param scale_vector: torch.Tensor, 缩放向量, scalar
        """
        # 缩放
        self.vertices = self.vertices_base.clone()
        self.vertices *= scale_vector
        # 旋转
        vert_numpy = self.vertices.cpu().numpy()
        rotate_vector_numpy = rotate_vector.cpu().numpy()
        self.vertices = R.from_rotvec(rotate_vector_numpy).apply(vert_numpy - get_mesh_center(vert_numpy)) + get_mesh_center(vert_numpy)
        self.vertices = torch.tensor(self.vertices).cuda().to(torch.float32)
        # 移动
        self.vertices += move_vector



class Scene:
    def __init__(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        self.objs = []
        self.data_dir = os.path.dirname(json_path)
        self.rot_num = 0
        self.move_num = 0
        self.obj_list_num = 0
        self.resize = False
        for obj_json in data["obj"]:
            if "mesh_dir" in obj_json or "trajectory" in obj_json:
                if "mesh_dir" in obj_json:
                    obj = ObjList(obj_json, self.data_dir)
                else:
                    obj = ObjAnim(obj_json, self.data_dir)
                self.obj_list_num += 1
            else:
                obj = Obj(obj_json, self.data_dir)
                if obj.rot_axis is not None:
                    self.rot_num += 1
                if obj.move_vec is not None:
                    self.move_num += 1
                if torch.any(obj.resize_base != 0):
                    self.resize = True
            self.objs.append(obj)
        solver_json = data["solver"]
        self.src_sample_num = solver_json["src_sample_num"]
        self.trg_sample_num = solver_json["trg_sample_num"]
        self.freq_min = solver_json["freq_min"]
        self.freq_max = solver_json["freq_max"]
        self.freq_min_log = np.log10(self.freq_min)
        self.freq_max_log = np.log10(self.freq_max)
        self.trg_pos_min = torch.tensor(solver_json["trg_pos_min"]).cuda()
        self.trg_pos_max = torch.tensor(solver_json["trg_pos_max"]).cuda()
        self.bbox_size = (self.trg_pos_max - self.trg_pos_min).max()
        self.bbox_center = (self.trg_pos_max + self.trg_pos_min) / 2
        self.trg_points = None


    def my_sample(self, max_resize=2, log=False,
                  sound_source: str="phone.obj",
                  seed: int=0,
                  freq_idx: int=0,
                  max_freq_idx: int=50,
                  ):
        """
        一个较为简单的取样，在场景内随机放置物体，随机大小、随机位置。
        :param sound_source: str, 声源物体的名称。该物体不会被移动或放大缩小。
        """
        RESIZE_BOUNDS = [1., 2.]
        MOVE_BOUNDS = [-0.2, 0.2]
        
        torch.manual_seed(seed)
        np.random.seed(int(seed * 100000) % 1000000007)
        
        for obj in self.objs:
            if obj.name == sound_source:
                # 声源物体不会被移动或放大缩小
                rotate_vector = torch.zeros(3).cuda()
                move_vector = torch.zeros(3).cuda()
                resize_vector = torch.ones(1).cuda()
            else:
                # 非声源物体随机移动、旋转、缩放
                rotate_vector = torch.rand(3).cuda() * 100000
                move_vector = torch.rand(3).cuda() * (MOVE_BOUNDS[1] - MOVE_BOUNDS[0]) + MOVE_BOUNDS[0]
                resize_vector = torch.rand(1).cuda() * (RESIZE_BOUNDS[1] - RESIZE_BOUNDS[0]) + RESIZE_BOUNDS[0]
            # 应用变换
            obj.move_the_object(move_vector, rotate_vector, resize_vector)
        
        # TODO: 碰撞检测
        
        
        # 组装场景
        self.vertices = torch.zeros(0, 3).cuda().to(torch.float32)
        self.triangles = torch.zeros(0, 3).cuda().to(torch.int32)
        self.neumann = torch.zeros(0).cuda().to(torch.complex64)
        for obj in self.objs:
            self.triangles = torch.cat(
                [self.triangles, obj.triangles + len(self.vertices)]
            )
          #  breakpoint()
            self.vertices = torch.cat([self.vertices, obj.vertices])
            self.neumann = torch.cat([self.neumann, obj.neumann])
        # 保证顶点和三角形的顺序是连续的
        self.vertices = self.vertices.contiguous().float()
        self.triangles = self.triangles.contiguous().int()
        # 随机选择频率
        self.freq_factor = torch.tensor(freq_idx / max_freq_idx).cuda()
        freq_log = (
            self.freq_factor * (self.freq_max_log - self.freq_min_log)
            + self.freq_min_log
        )
        freq = 10**freq_log
        self.k = (2 * np.pi * freq / 343.2).item()



    def sample(self, max_resize=2, log=False):
        '''
        Sample the scene: Randomly resize, rotate and move the objects for once and
        randomly select a frequency.

        For each object transition, a random factor is sampled from a uniform distribution.
        '''
        rot_factors = torch.rand(self.rot_num).cuda()
        move_factors = torch.rand(self.move_num).cuda()
        obj_list_factors = torch.rand(self.obj_list_num).cuda()
        #resize_factor = torch.rand(1).cuda()
        resize_factor = torch.ones(1).cuda() * 0.5
        #freq_factor = torch.rand(1).cuda()
        freq_factor = torch.ones(1).cuda() * 0.5
        rot_idx = 0
        move_idx = 0
        obj_list_idx = 0
        for obj in self.objs:
            if isinstance(obj, ObjList):
                obj.sample(obj_list_factors[obj_list_idx].item())
                obj_list_idx += 1
            if isinstance(obj, ObjAnim):
                obj.sample(obj_list_factors[obj_list_idx].item())
                obj_list_idx += 1
            obj.reset()
            obj.resize(resize_factor.item() * (max_resize - 1) + 1)
            if self.rot_num > 0:
                obj.rotation(rot_factors[rot_idx].item())
                if log and obj.rot_axis is not None:
                    print(obj.name, "rotaion idx:", rot_idx)
                if obj.rot_axis is not None and rot_idx < self.rot_num - 1:
                    rot_idx += 1
            if self.move_num > 0:
                if obj.name != "phone.obj": 
                    obj.move(move_factors[move_idx].item())
                    if log and obj.move_vec is not None:
                        print(obj.name, "move idx:", move_idx)
                    if obj.move_vec is not None and move_idx < self.move_num - 1:
                        move_idx += 1
            obj.shift()
        self.vertices = torch.zeros(0, 3).cuda().to(torch.float32)
        self.triangles = torch.zeros(0, 3).cuda().to(torch.int32)
        self.neumann = torch.zeros(0).cuda().to(torch.complex64)
        for obj in self.objs:
            self.triangles = torch.cat(
                [self.triangles, obj.triangles + len(self.vertices)]
            )
            self.vertices = torch.cat([self.vertices, obj.vertices])
            self.neumann = torch.cat([self.neumann, obj.neumann])
        self.vertices = self.vertices.contiguous().float()
        self.triangles = self.triangles.contiguous().int()
        self.rot_factors = rot_factors
        self.move_factors = move_factors
        self.obj_list_factors = obj_list_factors
        self.resize_factor = resize_factor
        self.freq_factor = freq_factor
        freq_log = (
            self.freq_factor * (self.freq_max_log - self.freq_min_log)
            + self.freq_min_log
        )
        freq = 10**freq_log
        self.k = (2 * np.pi * freq / 343.2).item()

    def setting(self, rot_factors:torch.Tensor=None, move_factors:torch.Tensor=None, 
                obj_list_factors:torch.Tensor=None, resize_factor:torch.Tensor=None, 
                freq_factor:torch.Tensor=None):
        '''
        Manually set the scene parameters. Instead of randomly sampling, the scene is set by the given factors.

        We suggest to define the frequency manually, as the frequency is the main parameter of the scene. Otherwise, the frequency will be randomly sampled.
        '''

        if rot_factors:
            assert self.rot_num == len(rot_factors), "rot_factors length error"
        if move_factors:    
            assert self.move_num == len(move_factors), "move_factors length error"
        if obj_list_factors:
            assert self.obj_list_num == len(obj_list_factors), "obj_list_factors length error"
        if resize_factor:
            assert self.resize, "resize_factor error"
        
        if freq_factor.item() == 0:
            self.freq_factor = torch.rand(1).cuda()
        else :
            self.freq_factor = freq_factor

        freq_log = (
            self.freq_factor * (self.freq_max_log - self.freq_min_log)
            + self.freq_min_log
        )
        freq = 10**freq_log

        self.k = (2 * np.pi * freq / 343.2).item()

        rot_idx = 0
        move_idx = 0
        obj_list_idx = 0
        for obj in self.objs:
            if isinstance(obj, ObjList):
                obj.sample(obj_list_factors[obj_list_idx].item())
                obj_list_idx += 1
            if isinstance(obj, ObjAnim):
                obj.sample(obj_list_factors[obj_list_idx].item())
                obj_list_idx += 1
            obj.reset()
            if resize_factor:
                obj.resize(resize_factor.item())
            if self.rot_num > 0:
                obj.rotation(rot_factors[rot_idx].item())
                if obj.rot_axis is not None and rot_idx < self.rot_num - 1:
                    rot_idx += 1
            if self.move_num > 0:
                obj.move(move_factors[move_idx].item())
                if obj.move_vec is not None and move_idx < self.move_num - 1:
                    move_idx += 1
            obj.shift()

        self.vertices = torch.zeros(0, 3).cuda().to(torch.float32)
        self.triangles = torch.zeros(0, 3).cuda().to(torch.int32)
        self.neumann = torch.zeros(0).cuda().to(torch.complex64)
        for obj in self.objs:
            self.triangles = torch.cat(
                [self.triangles, obj.triangles + len(self.vertices)]
            )
            self.vertices = torch.cat([self.vertices, obj.vertices])
            self.neumann = torch.cat([self.neumann, obj.neumann])
        self.vertices = self.vertices.contiguous().float()
        self.triangles = self.triangles.contiguous().int()

        self.rot_factors = rot_factors
        self.move_factors = move_factors
        self.obj_list_factors = obj_list_factors
        self.resize_factor = resize_factor


    def solve(self, man_trg_factor=None):
        '''
        Solve the scene: Calculate the potential of the current (sampled) scene at the target points.
        
        If `man_trg_point` is `None`, then the function will 
        ramdomly sample target points in the bounding box using spherical coordinates for `self.trg_sample_num` times.
        
        Returns:
        `self.potential`: a tensor of shape `(len(self.trg_points), 1)`, which is the potential of the scene at the target points.
        '''
        solver = BEM_Solver(self.vertices, self.triangles)
        self.dirichlet = solver.neumann2dirichlet(self.k, self.neumann)

        if man_trg_factor is None:
            sample_points_base = torch.rand(
                self.trg_sample_num, 3, device="cuda", dtype=torch.float32
            )
            self.trg_factor = sample_points_base
        else :
            self.trg_factor = man_trg_factor
            # self.trg_sample_num = man_trg_factor.shape[0]
        
        # rs = (sample_points_base[:, 0] + 1) * self.bbox_size
        rs = (self.trg_factor[:, 0]) * self.bbox_size * 2 # (0,2) times the bounding box size
        theta = self.trg_factor[:, 1] * np.pi # (0, pi)
        phi = self.trg_factor[:, 2] * 2 * np.pi - np.pi # (-pi, pi)
        xs = rs * torch.sin(theta) * torch.cos(phi)
        ys = rs * torch.sin(theta) * torch.sin(phi)
        zs = rs * torch.cos(theta)
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.trg_points = torch.stack([xs, ys, zs], dim=-1) + self.bbox_center

        self.potential = solver.boundary2potential(
            self.k, self.neumann, self.dirichlet, self.trg_points
        ).cpu()

    def show(self, logged_values=False):
        # logged_values: bool, if True, the values will take a log transformation
        if logged_values == False:
            vis = Visualizer()
            vis.add_mesh(self.vertices, self.triangles, torch.log10(torch.clip(self.neumann.abs(), 1e-6, 1e6)))
            if self.trg_points is not None:
                vis.add_points(self.trg_points, torch.log10(torch.clip(self.potential.abs(), 1e-6, 1e6)))
            vis.show()


class EditableModalSound:

    def __init__(self, data_dir, ffat_res=(64, 32), uniform=False):
        with open(f"{data_dir}/config.json", "r") as file:
            js = json.load(file)
            sample_config = js.get("sample", {})
            obj_config = js.get("vibration_obj", {})
            self.size_base = obj_config.get("size")

        data = torch.load(f"{data_dir}/modal_data.pt")
        self.vertices_base = data["vertices"]
        self.triangles = data["triangles"]
        self.neumann_vtx = data["neumann_vtx"]
        self.ks_base = data["ks"]
        self.mode_num = len(self.ks_base)

        self.freq_rate = sample_config.get("freq_rate")
        self.size_rate = sample_config.get("size_rate")
        self.bbox_rate = sample_config.get("bbox_rate")
        self.sample_num = sample_config.get("sample_num")
        self.point_num_per_sample = sample_config.get("point_num_per_sample")

        self.ffat_res = ffat_res
        xs = torch.linspace(0, 1, ffat_res[0], device="cuda", dtype=torch.float32)
        ys = torch.linspace(0, 1, ffat_res[1], device="cuda", dtype=torch.float32)
        self.gridx, self.gridy = torch.meshgrid(xs, ys)
        self.uniform = uniform
        if uniform:
            self.point_num_per_sample = ffat_res[0] * ffat_res[1]

    def sample(self, freqK_base=None, sizeK_base=None):
        if freqK_base is None:
            self.freqK_base = torch.rand(1).cuda()
        else:
            self.freqK_base = freqK_base

        if sizeK_base is None:
            self.sizeK_base = torch.rand(1).cuda()
        else:
            self.sizeK_base = sizeK_base

        self.freqK = self.freqK_base * self.freq_rate
        self.sizeK = 1.0 / (1 + self.sizeK_base * (self.size_rate - 1))

        self.vertices = self.vertices_base * self.sizeK
        self.ks = self.ks_base * self.freqK / self.sizeK**0.5

        if self.uniform:
            sample_points_base = torch.zeros(
                self.ffat_res[0] * self.ffat_res[1],
                3,
                device="cuda",
                dtype=torch.float32,
            )
            sample_points_base[:, 0] = 0.5
            sample_points_base[:, 1] = self.gridx.reshape(-1)
            sample_points_base[:, 2] = self.gridy.reshape(-1)
        else:
            sample_points_base = torch.rand(
                self.point_num_per_sample, 3, device="cuda", dtype=torch.float32
            )

        self.sample_points_base = sample_points_base
        rs = (
            (sample_points_base[:, 0] * (self.bbox_rate - 1) + 1) * self.size_base * 0.7
        )
        theta = sample_points_base[:, 1] * 2 * np.pi - np.pi
        phi = sample_points_base[:, 2] * np.pi
        xs = rs * torch.sin(phi) * torch.cos(theta)
        ys = rs * torch.sin(phi) * torch.sin(theta)
        zs = rs * torch.cos(phi)
        self.trg_points = torch.stack([xs, ys, zs], dim=-1)

        input_x = torch.zeros(
            self.sample_points_base.shape[0],
            3 + 1 + 1,
            dtype=torch.float32,
        )
        input_x[:, :3] = self.sample_points_base
        input_x[:, 3] = self.sizeK_base
        input_x[:, 4] = self.freqK_base
        return input_x

    def solve(self):
        bem_solver = BEM_Solver(self.vertices, self.triangles)
        potentials = []
        for i in range(self.mode_num):
            dirichlet_vtx = bem_solver.neumann2dirichlet(
                self.ks[i].item(), self.neumann_vtx[i]
            )
            potential = bem_solver.boundary2potential(
                self.ks[i].item(), self.neumann_vtx[i], dirichlet_vtx, self.trg_points
            )
            potentials.append(potential)

        self.potentials = torch.stack(potentials, dim=0).cpu()
        return self.potentials

    def show(self, mode_idx=0):
        vis = Visualizer()
        vis.add_mesh(self.vertices, self.triangles, self.neumann_vtx[mode_idx].abs())
        vis.add_points(self.trg_points, self.potentials[mode_idx].abs())
        vis.show()

def initial_config(data_dir, src_sample_num : int = 1, trg_sample_num: int = 1000,
                    freq_min: int = 100, freq_max: int = 10000, 
                    trg_pos_min: list = [-.5,-.5,-.5], trg_pos_max: list = [.5,.5,.5]):
    '''
    Initialize the necessary scene dataset file -- config.json:

    - data_dir: str, the directory of the scene data, i.e. "dataset/scene_name"
    - src_sample_num: int, default number of source samples, i.e. number of times the scene is sampled
    - trg_sample_num: int, default number of target samples, randomly sampled in bounding box space
    - freq_min, freq_max: int, minimum and maximum frequency of the sound
    - trg_pos_min, trg_pos_max: list, minimum and maximum target position of the sound, i.e. the bounding box of the scene
      - in the shape of [x_min, y_min, z_min] and [x_max, y_max, z_max]

    '''
    config = {
        "obj": [],
        "solver": {
            "src_sample_num": 10,
            "trg_sample_num": 1000,
            "freq_min": 100,
            "freq_max": 1000,
            "trg_pos_min": [-1, -1, -1],
            "trg_pos_max": [1, 1, 1],
        },
    } # a sample config.json

    with open(f"{data_dir}/config.json", "w") as file:
        json.dump(config, file)

    return data_dir


def config_add_obj(data_dir, obj_name, size, resize = None, rot_axis=None, rot_pos=None, rot_max_deg:float=None, move=None, position=None, vibration=None):
    '''
    Add an object to the scene config.json.

    Required arguments:
        - data_dir: str, the directory of the scene data
        - obj_name: str, the name of the mesh object file, i.e. "obj.obj"
        - size: float, the size of the object. Our Scene class will normalize the object to this size.

    Optional arguments: when not specified, the object will not be resized, rotated, moved or vibrated.
        - resize: list, the resize factor of the object, i.e. [0,0,1], which means no resize in x and y, and resize in z by 1x`max_resize` scale.
        - rot_axis: list, the rotation axis of the object, i.e. [1,0,0] for x-axis
        - rot_pos: list, the rotation axis position of the object, i.e. [0,0,0] for the origin
        - rot_max_deg: float, the maximum rotation degree of the object
        - move: list, the move vector of the object, i.e. [0,0,0] for no move
        - position: list, the position of the object center
        - vibration: str, the vibration direction of the object, i.e. "-x" for vibrating in the negative x direction


    Notice: The mesh should be in .obj format and placed in the data_dir.

    '''
    with open(f"{data_dir}/config.json", "r") as file:
        config = json.load(file)
    
    obj = {
        "mesh": obj_name,
        "size": size,
    }

    if resize is not None:
        obj["resize"] = resize

    if rot_axis is not None and rot_pos is not None and rot_max_deg is not None:
        obj["rot_axis"] = rot_axis
        obj["rot_pos"] = rot_pos
        obj["rot_max_deg"] = rot_max_deg
    
    if move is not None:
        obj["move"] = move
    
    if position is not None:
        obj["position"] = position
    else :
        obj["position"] = [0,0,0]

    if vibration is not None:
        obj["vibration"] = vibration
    
    config["obj"].append(obj)

    with open(f"{data_dir}/config.json", "w") as file:
        json.dump(config, file)
    

def initial_files(data_dir):
    ''' 
    TODO
    '''
    pass
    with open(f"{data_dir}/config.json", "w") as file:
        json.dump(config, file)

    with open(f"{data_dir}/animation_generator.py", "w") as file:
        file.write("# please generate your animation")

    np.savez(f"{data_dir}/animation_data.npz", data={'x' : np.zeros(0), 'fps' : 30})



def genarate_sample_scene(data_dir, data_name, src_sample_num = None, trg_sample_num = None , show_scene:bool=False):
    '''
    Generate the scene data and save it to the data_dir/data/data_name.pt
    '''
    scene = Scene(f"{data_dir}/config.json")

    if src_sample_num is None:
        src_sample_num = scene.src_sample_num
    if trg_sample_num is None:
        trg_sample_num = scene.trg_sample_num

    x = torch.zeros(
        src_sample_num,
        trg_sample_num,
        3 + scene.rot_num + scene.move_num + (1 if scene.resize else 0) + 1,
        dtype=torch.float32,
    )
    y = torch.zeros(src_sample_num, trg_sample_num, 1, dtype=torch.float32)

    for src_idx in tqdm(range(src_sample_num)):
        scene.sample()
        scene.solve()
        x[src_idx, :, :3] = scene.trg_factor
        if scene.rot_num > 0:
            x[src_idx, :, 3 : 3 + scene.rot_num] = scene.rot_factors
        if scene.move_num > 0:
            x[src_idx, :, 3 + scene.rot_num : 3 + scene.rot_num + scene.move_num] = (
                scene.move_factors
            )
        if scene.resize:
            x[src_idx, :, -2] = scene.resize_factor
        x[src_idx, :, -1] = scene.freq_factor
        y[src_idx] = scene.potential.abs().unsqueeze(-1)

        if src_idx == 0 and show_scene:
            scene.show()

    os.makedirs(f"{data_dir}/data", exist_ok=True)
    torch.save({"x": x, "y": y}, f"{data_dir}/data/{data_name}.pt")
    return x, y


def generate_sample_scene_simpler(data_dir, data_name, src_sample_num = None, trg_sample_num = None , show_scene:bool=False):
    '''
    一个简化版的generate_sample_scene。
    对每个场景，只生成一个样本，并存储相应的数据以及几何形状。
    注意，此代码使用 cuda，而且基本能吃满整个GPU，所以没有多进程的必要。
    '''
    scene = Scene(f"{data_dir}/config.json")
    
    if src_sample_num is None:
        src_sample_num = scene.src_sample_num
    if trg_sample_num is None:
        trg_sample_num = scene.trg_sample_num
        
    for src_idx in tqdm(range(src_sample_num)):
        import time
        seed = time.time()
        x = torch.zeros(
                trg_sample_num, 3,
                dtype=torch.float32,
            )
        y = torch.zeros(trg_sample_num, 65, dtype=torch.float32)
        for freq_idx in tqdm(range(65)):
            scene.my_sample(seed=seed, freq_idx=freq_idx, max_freq_idx=65)
            scene.solve()
            x[:, :3] = scene.trg_points
         #   x[:, :3] = scene.trg_points#scene.trg_factor
         #   x[:, -1] = scene.freq_factor
         
            y[:, freq_idx] = scene.potential.abs()#.unsqueeze(-1)

         #   if src_idx != 0 and show_scene:
         #       scene.show()
                
        # 存储x，y以及几何形状  trg_points
        torch.save({"x": x, "y": y}, f"{data_dir}/data/train_data/{data_name}_{src_idx}.pt")
        # 以obj格式存储几何形状
        import trimesh
        mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
        mesh.export(f"{data_dir}/data/train_mesh/{data_name}_{src_idx}.obj")