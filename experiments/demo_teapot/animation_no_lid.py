import sys, os
sys.path.append("./")
# os.environ["TORCH_CUDA_ARCH_LIST"]="8.0"

os.environ["CUDA_VISIBLE_DEVICES"]="7"

import torch
import numpy as np
import trimesh
from tqdm import tqdm
import seaborn as sns
from scipy.io import wavfile

from src.scene import Scene
from src.utils import *

'''
对齐新版本scene采样逻辑的 teapot demo 动画声音生成
调试：删除盖子有关的动画，仅移动视角
'''
data_dir = "dataset/teapot"

container = trimesh.load_mesh(f"{data_dir}/hushen.obj")
# plane = trimesh.load_mesh(f"{data_dir}/gaizi.obj")
ball = trimesh.load_mesh(f"{data_dir}/ball.obj")
ear = trimesh.load_mesh(f"{data_dir}/ear.obj")

BALL_SCALE = 0.03
# normalize & scale
container = container.apply_scale(0.28 / np.max(container.extents))
container.apply_translation(-(container.vertices.max(axis=0)+container.vertices.min(axis=0))/2)
# plane = plane.apply_scale(PLANE_SCALE / np.max(plane.extents))
ball = ball.apply_scale(BALL_SCALE / np.max(ball.extents))
ear = ear.apply_scale(0.05 / np.max(ear.extents))

# plane_trans = plane.centroid + np.array([0, 0.035, -0.01])

# plane = plane.apply_translation(-plane.centroid)
# plane = plane.apply_scale(1.08)
# PLANE_SCALE = np.max(plane.extents)
# centralize
ball.apply_translation(-(ball.vertices.max(axis=0)+ball.vertices.min(axis=0))/2)
ear.apply_translation(-(ear.vertices.max(axis=0)+ear.vertices.min(axis=0))/2)

######## 动画部分 ##########
'''
计算除小球外（因其固定在原点）所有部分的变换四元阵，第二部分组装时使用
'''
fps = 30
frame_num = fps * 12

ani_container = np.zeros((frame_num, 4, 4))
# ani_plane = np.zeros((frame_num, 4, 4))
ani_listener = np.zeros((frame_num, 4, 4))

# 单位变换矩阵
identity_matrix = np.eye(4)

ani_container[:,] = identity_matrix
ani_listener[:,] = identity_matrix
# ani_plane[:,] = identity_matrix

def transition_matrix(angle=0, axis=[1,0,0], translation=[0,0,0], point=None):
    """
    计算旋转矩阵和平移矩阵的组合
    :param angle: 旋转角度
    :param axis: 旋转轴
    :param translation: 平移向量
    :return: 4x4的变换矩阵
    """
    rotation = trimesh.transformations.rotation_matrix(angle, axis, point)
    translation = trimesh.transformations.translation_matrix(translation)
    return np.dot(translation, rotation)

# 计算某参数的三阶贝叶斯曲线
'''
def cubic_bezier(start, end, frame_num):
    """计算三次贝塞尔曲线的点"""
    p0, p1, p2, p3 = 0, 0.00001, 0.99999, 1  # 控制点
    ts = np.linspace(start, end, num=frame_num)
    return [
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
        for t in ts
    ]
    '''
def bayesian_smooth(begin, end, frame_num):
    # t = np.linspace(0, 1, frame_num)
    # return (1-t)**3 * begin + 3*(1-t)**2 * t * begin + 3*(1-t) * t**2 * end + t**3 * end
    t = np.linspace(0, 1, frame_num)
    p0, p1, p2, p3 = 0, 0.00001, 0.99999, 1  # 控制点
    return ((1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3)* (end - begin) + begin

def save_coordinates_to_npz(file_path, coordinates):
    np.savez(file_path, **coordinates)

# print("check bayesian smooth", bayesian_smooth(0, 1, 11))

# 初始translation位置, 保持2秒
LISTENER_POS = [0.8, 0, 0]

# debug动画：仅移动listener，3秒-9秒以[0, 0, 0]为中心在x-y平面做圆周运动
RADIUS = 0.8

listener_translate = LISTENER_POS
ani_listener[0:fps*2] = transition_matrix(0, [0, 0, 1], listener_translate)
ani_container[0:fps*15] = transition_matrix()

# 3 seconds: Listener moves in a circle (2*fps to 5*fps-1)
theta = bayesian_smooth(0, 2 * np.pi, fps*3)
for i in range(fps*2, fps*5):
    x = RADIUS * np.cos(theta[i-fps*3])
    y = RADIUS * np.sin(theta[i-fps*3])
    listener_translate = [x, y, LISTENER_POS[2]]
    ani_listener[i] = transition_matrix(0, [0, 0, 1], listener_translate)


# 2 seconds: Static (5*fps to 7*fps-1)
ani_listener[fps*5:fps*7] = transition_matrix(0, [0, 0, 1], listener_translate)
# 3 seconds: Listener moves back in a circle (7*fps to 10*fps-1)
for i in range(fps*7, fps*10):
    x = RADIUS * np.cos(-theta[i-fps*7])
    y = RADIUS * np.sin(-theta[i-fps*7])
    listener_translate = [x, y, LISTENER_POS[2]]
    ani_listener[i] = transition_matrix(0, [0, 0, 1], listener_translate)
# 2 seconds: Static (10*fps to 12*fps-1)
ani_listener[fps*10:fps*12] = transition_matrix(0, [0, 0, 1], listener_translate)


######## 组装每一帧场景并生成声音 ##########

# scene求解函数
def eval_listener(scene:Scene, x, freq_pos):
    """
    计算x序列的声音。调用scene的sample和solve方法。
    return: (len(x), len(freq_pos)) torch.Tensor
    """
    x = torch.tensor(x, device="cuda", dtype=torch.float32).reshape(-1,3)
    nc_spec = None
    for freq_fac in tqdm(range(len(freq_pos)), position=1, desc=f"Solving {len(freq_pos)} freqs", leave=False):
        scene.enclose_sample(sound_source="ball.obj", 
            freq_idx=freq_fac, _transition=False, _resize=False, _rotate=False)
        scene.solve(x)
        
        if torch.isnan(scene.potential).any():
            print(f"")
            print(f"")
            nan_indices = torch.isnan(scene.potential).nonzero(as_tuple=True)
            print(f"Warning: NaN in scene.potential at freq {freq_fac}")
            print(f"example nan x: {x}")
            # 也许是求解器随机性导致的错误，重新计算该循环
            cnt = 0
            while torch.isnan(scene.potential).any():
                scene.enclose_sample(sound_source="ball.obj", 
                    freq_idx=freq_fac, _transition=False, _resize=False, _rotate=False)
                scene.solve(x)
                cnt += 1
                if cnt > 10:
                    scene.potential[nan_indices] = 1e-6
                    print(f"Warning: NaN in scene.potential at freq {freq_fac} after 10 tries")
                    break
            print("")
                    
            # scene.potential = torch.zeros_like(scene.potential)
        if nc_spec is None:
            nc_spec = scene.potential.abs().unsqueeze(-1)
        else:
            nc_spec = torch.cat([nc_spec, scene.potential.abs().unsqueeze(-1)], dim=-1)

    nc_spec = nc_spec.reshape(x.shape[0], -1)
    return nc_spec


sampling_rate = 16000
n_fft = 128
save_fig = True
stereo = False
audio_name = "no_lid.wav"

# 是否加载已有的spec
LOAD_SPEC = False

scene = Scene(f"{data_dir}/config.json", normalize=False)

freq_bins = calculate_bin_frequencies(n_fft, sampling_rate=sampling_rate)

freq_pos = torch.zeros(len(freq_bins), device="cuda", dtype=torch.float32)
for freq_i in tqdm(range(len(freq_bins)), desc="Calculating freq_factor"):
    freq_bin = freq_bins[freq_i]
    if freq_bin < scene.freq_min:
        continue
    if freq_bin > scene.freq_max:
        freq_pos[freq_i] = 1
        continue
    freq_pos[freq_i] = (np.log10(freq_bin) - scene.freq_min_log) / (
        scene.freq_max_log - scene.freq_min_log
    )

nc_spec = None
os.makedirs(f"{data_dir}/meshes/animation", exist_ok=True)
os.makedirs(f"{data_dir}/meshes/visualize", exist_ok=True)
os.makedirs(f"{data_dir}/spec_figs", exist_ok=True)

old_ob = None
old_listener = None

if LOAD_SPEC is False:
    sampling_points = ani_listener[:, :3, 3].reshape(-1, 3)
    x = torch.from_numpy(sampling_points).float().cuda()
    
    container_matrix = ani_container[0]
    container_moved = container.copy()
    container_moved.apply_transform(container_matrix)

    obstacle = container_moved
    # 覆盖到obstacle.obj
    obstacle.export(f"{data_dir}/obstacle.obj")
    # 保存到meshes，编号长度固定为4位
    obstacle = trimesh.util.concatenate([obstacle, ball])

    scene = Scene(f"{data_dir}/config.json", normalize=False)
    # 获取listener位置
    spec = eval_listener(scene, x, freq_pos)
    
    if nc_spec is None:
        nc_spec = spec
    else:
        nc_spec = torch.cat([nc_spec, spec], dim=0) # (frame_num, freq_num)

    print(f"nc_spec shape: {nc_spec.shape}")

    nc_spec = nc_spec.cpu().numpy().T
    np.save(f"{data_dir}/nc_spec", nc_spec)
else:
    nc_spec = np.load(f"{data_dir}/nc_spec.npy")

nc_spec = np.clip(nc_spec, 0, 0.1)

import matplotlib.pyplot as plt

if save_fig:
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log10(nc_spec), cmap="viridis", )
    plt.title("Heatmap of nc_spec in log")
    plt.ylabel("Frequency Bins")
    plt.xlabel("Source Number")
    plt.savefig(f"{data_dir}/nc_spec_heatmap.png")
    plt.close()

src_num = frame_num

os.makedirs(f"{data_dir}/spectrogram", exist_ok=True)

if stereo:
    audio = apply_spec_mask_to_audio_stereo(
        1, nc_spec, src_num, n_fft=n_fft, animation_frame_rate=fps, trg_sample_rate=sampling_rate,
        save_spec=True, save_path=f"{data_dir}/spectrogram"
    )
else:
    audio = apply_spec_mask_to_audio(
        1, nc_spec, src_num, n_fft=n_fft, animation_frame_rate=fps, trg_sample_rate=sampling_rate,
        save_spec=True, save_path=f"{data_dir}/spectrogram"
    )

wavfile.write(f"{data_dir}/{audio_name}", sampling_rate, audio)

#### debug ####
DEBUG = False # <--- 设置为 True 来启用调试
dbg_frame = 0 # <--- 设置需要调试的帧
dbg_freq_idx = 5 # <--- 选择一个频率进行可视化 (例如中间频率)

# 可视化该帧，y平面上的场切片图
if DEBUG:
    import matplotlib.pyplot as plt
    print(f"--- Debugging Frame {dbg_frame} ---")

    # 1. 重建该帧的场景几何
    container_matrix = ani_container[dbg_frame]

    container_moved = container.copy()
    container_moved.apply_transform(container_matrix)
    # 组合障碍物 (不包括声源球，因为它在原点，BEM求解需要)
    obstacle_dbg = container_moved
    # 保存该帧的障碍物网格以供Scene加载
    dbg_obstacle_path = f"{data_dir}/obstacle.obj"
    print(f"Saving debug obstacle mesh to: {dbg_obstacle_path}")
    obstacle_dbg.export(dbg_obstacle_path)

    # 计算包围盒和采样距离
    bounding_box = obstacle_dbg.bounds
    center = [0,0,0]
    diag_length = np.linalg.norm(obstacle_dbg.extents)
    # 设定采样平面y坐标和采样范围
    y_slice_coord = center[1] # 在质心高度切片
    # y_slice_coord = diag_length * 1.6 # 或者选择一个固定高度，例如容器高度的一半以上
    # 采样范围至少覆盖包围盒，并向外扩展3个对角线长度
    sampling_extent = max(obstacle_dbg.extents) + diag_length*2
    grid_res = 200 # 采样分辨率

    print(f"Obstacle BBox: {bounding_box}")
    print(f"Obstacle Centroid: {center}")
    print(f"Obstacle Diagonal Length: {diag_length:.4f}")
    print(f"Sampling Plane Y: {y_slice_coord:.4f}")
    print(f"Sampling Extent (half-side): {sampling_extent:.4f}")
    print(f"Sampling Grid Resolution: {grid_res}x{grid_res}")

    # # 2. 创建采样点网格 (Z-Y平面)
    y_coords = np.linspace(center[1] - sampling_extent, center[1] + sampling_extent, grid_res)
    z_coords = np.linspace(center[2] - sampling_extent, center[2] + sampling_extent, grid_res)
    yy, zz = np.meshgrid(y_coords, z_coords)
    sampling_points = np.vstack([np.full(yy.size, y_slice_coord), yy.ravel(), zz.ravel()]).T
    sampling_points_torch = torch.from_numpy(sampling_points).float().cuda()

    # 2debug. 创建采样点网格（x-y-z）
    # x_coords = np.linspace(center[0] - sampling_extent, center[0] + sampling_extent, grid_res)
    # y_coords = np.linspace(center[1] - sampling_extent, center[1] + sampling_extent, grid_res)
    # z_coords = np.linspace(center[2] - sampling_extent, center[2] + sampling_extent, grid_res)
    # xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords)
    # sampling_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    # sampling_points_torch = torch.from_numpy(sampling_points).float().cuda()

    # 3. 初始化Scene并计算选定频率下的声场
    print(f"Loading scene for frame {dbg_frame}:")
    # 使用特定帧的障碍物文件初始化Scene
    scene_dbg = Scene(f"{data_dir}/config.json", normalize=False)

    print(f"Calculating potential for frequency index {dbg_freq_idx} ({freq_bins[dbg_freq_idx]:.2f} Hz)")
    # 注意：这里假设声源 'ball.obj' 位于原点，与Scene配置一致
    scene_dbg.enclose_sample(sound_source="ball.obj",
                           freq_idx=dbg_freq_idx, _transition=False, _resize=False, _rotate=False)
    scene_dbg.solve(sampling_points_torch)
    scene_dbg.show()
    # scene_dbg.solve()

    # 处理潜在的NaN值 (与eval_listener类似)
    if torch.isnan(scene_dbg.potential).any():
        nan_indices = torch.isnan(scene_dbg.potential).nonzero(as_tuple=True)
        print(f"Warning: NaN detected in potential calculation for debug frame.")
        # 尝试重新计算
        cnt = 0
        while torch.isnan(scene_dbg.potential).any() and cnt < 5:
             print(f"Retrying potential calculation... (Attempt {cnt+1})")
             scene_dbg.enclose_sample(sound_source="ball.obj",
                                   freq_idx=dbg_freq_idx, _transition=False, _resize=False, _rotate=False)
             scene_dbg.solve(sampling_points_torch)
             cnt += 1
        if torch.isnan(scene_dbg.potential).any():
            print(f"Warning: NaN persists. Replacing NaNs with 1e-9.")
            scene_dbg.potential = torch.nan_to_num(scene_dbg.potential, nan=1e-9) # 使用小值替换NaN

    potential_slice = scene_dbg.potential.abs().cpu().numpy()
    # if scene_dbg.rand_index is not None: # 去掉，期望不发生问题
    #     index = scene_dbg.rand_index.cpu().numpy()
    #     potential_slice[index] = potential_slice
        
    potential_slice = potential_slice.reshape(grid_res, grid_res)

    # 4. 打印部分结果
    print("Potential slice shape:", potential_slice.shape)
    print("Potential slice (center 5x5 values):")
    center_idx = grid_res // 2
    print(potential_slice[center_idx-2:center_idx+3, center_idx-2:center_idx+3])
    print(f"Potential Min: {np.min(potential_slice):.4e}, Max: {np.max(potential_slice):.4e}, Mean: {np.mean(potential_slice):.4e}")


    # 5. 可视化场切片图和障碍物截面
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制声压幅值热力图 (使用对数刻度可能效果更好)
    im = ax.pcolormesh(zz, yy, np.log10(potential_slice + 1e-9),vmax=-2, vmin=-7, shading='gouraud', cmap='viridis') # 加小量避免log(0)
    fig.colorbar(im, ax=ax, label='Log10(Potential Magnitude)')

    # 计算并绘制障碍物在该平面上的截面
    # TODO


    ax.set_xlabel("Z coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Acoustic Potential Magnitude Slice at Y={y_slice_coord:.3f}\nFrame={dbg_frame}, Freq={freq_bins[dbg_freq_idx]:.1f} Hz")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(z_coords.min(), z_coords.max())
    ax.set_ylim(y_coords.min(), y_coords.max())
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存图像
    dbg_fig_path = f"{data_dir}/debug_field_slice_frame_{dbg_frame}_freq_{dbg_freq_idx}.png"
    plt.savefig(dbg_fig_path)
    print(f"Debug visualization saved to: {dbg_fig_path}")
    plt.close(fig) # 关闭图像，避免显示出来

    # 清理临时障碍物文件
    # os.remove(dbg_obstacle_path)
    print(f"--- Debugging Frame {dbg_frame} Finished ---")
