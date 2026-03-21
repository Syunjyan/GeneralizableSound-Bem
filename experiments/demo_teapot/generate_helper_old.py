'''
2025-04-28
1.优先验证 数据集正确性（动画脚本）
2. 暂时固定声源位置，实验采样密度
3. 实验新采样方法（声源位置&场景位置同时随机）
'''
import sys

sys.path.append("./")

from src.scene import Scene, Obj
import torch
import os, sys
from tqdm import tqdm


import numpy as np
import trimesh


def transition_matrix(angle, axis, translation, point=None):
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

def concat_obstacle(data_dir, seed = None, plane_move = True, scene_move = True):
    '''
    障碍物的组装，平移、旋转采样在这里完成。
    '''
    container = trimesh.load_mesh(f"{data_dir}/hushen.obj")
    plane = trimesh.load_mesh(f"{data_dir}/gaizi.obj")
    # normalize
    container = container.apply_scale(0.26 / np.max(container.extents))
    container = container.apply_translation([0, -0.09, 0])
    # plane = plane.apply_scale(1 / np.max(plane.extents))

    # container_center = (container.bounds[0] + container.bounds[1]) / 2
    # container = container.apply_translation(-container_center)

    plane_center = (plane.bounds[0] + plane.bounds[1]) / 2
    plane_trans = plane_center + np.array([0, 0.02, 0])
    plane = plane.apply_translation(-plane_center)
    # rescale
    # CONTAINER_SCALE = 0.15
    # PLANE_SCALE = 0.11
    # container = container.apply_scale(CONTAINER_SCALE)
    # plane = plane.apply_scale(PLANE_SCALE)
    # plane_translate_z = CONTAINER_SCALE / 2 + 0.01
    # align
    container_scale = np.max(container.extents)
    plane_scale = np.max(plane.extents)

    if seed is None:
        seed = np.random.randint(1000000000)
    np.random.seed(seed)
    # randomly rotate
    if plane_move:
        theta = np.random.rand()*np.pi
    else:
        theta = 0
    plane_matrix = transition_matrix(-theta, [0,0,1], plane_trans, point=[plane_scale/2, 0, 0])
    plane.apply_transform(plane_matrix)
    
    if scene_move:
        pass

    # concate
    obstacle = trimesh.util.concatenate([container, plane])
    # return vert and tri
    return obstacle.vertices, obstacle.faces

def show_debug(gpu, scene_dbg: Scene, dbg_freq_idx):
    import matplotlib.pyplot as plt

    # 计算包围盒和采样距离
    center = [0,0,0]
    # 设定采样平面y坐标和采样范围
    x_slice_coord = center[0] # 在质心高度切片
    # 采样范围至少覆盖包围盒，并向外扩展3个对角线长度
    sampling_extent = 1.2
    grid_res = 200 # 采样分辨率

    print(f"Obstacle Centroid: {center}")
    print(f"Sampling Plane X: {x_slice_coord:.4f}")
    print(f"Sampling Extent (half-side): {sampling_extent:.4f}")
    print(f"Sampling Grid Resolution: {grid_res}x{grid_res}")

    # # 2. 创建采样点网格 (Y-Z平面)
    y_coords = np.linspace(center[1] - sampling_extent, center[1] + sampling_extent, grid_res)
    z_coords = np.linspace(center[2] - sampling_extent, center[2] + sampling_extent, grid_res)
    yy, zz = np.meshgrid(y_coords, z_coords)
    sampling_points = np.vstack([ np.full(yy.size, x_slice_coord),yy.ravel(), zz.ravel()]).T
    sampling_points_torch = torch.from_numpy(sampling_points).float().cuda()

    # 2debug. 创建采样点网格（x-y-z）
    # x_coords = np.linspace(center[0] - sampling_extent, center[0] + sampling_extent, grid_res)
    # y_coords = np.linspace(center[1] - sampling_extent, center[1] + sampling_extent, grid_res)
    # z_coords = np.linspace(center[2] - sampling_extent, center[2] + sampling_extent, grid_res)
    # xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords)
    # sampling_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    # sampling_points_torch = torch.from_numpy(sampling_points).float().cuda()
    freq_bins = scene_dbg.freq_bins

    print(f"Calculating potential for frequency index {dbg_freq_idx} ({freq_bins[dbg_freq_idx]:.2f} Hz)")
    # 注意：这里假设声源 'ball.obj' 位于原点，与Scene配置一致
    scene_dbg.enclose_sample(sound_source="ball.obj",
                           freq_idx=dbg_freq_idx, _transition=False, _resize=False, _rotate=False)
    scene_dbg.solve(sampling_points_torch)
    # scene_dbg.show()
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
    im = ax.pcolormesh(zz, yy, np.log10(potential_slice + 1e-9),vmax=-3.0, vmin=-5.5, shading='gouraud', cmap='viridis') # 加小量避免log(0)
    fig.colorbar(im, ax=ax, label='Log10(Potential Magnitude)')

    # 计算并绘制障碍物在该平面上的截面
    # TODO


    ax.set_ylabel("Y coordinate")
    ax.set_xlabel("Z coordinate")
    ax.set_title(f"Acoustic Potential Magnitude Slice at X={x_slice_coord:.3f}\n, Freq={freq_bins[dbg_freq_idx]:.1f} Hz")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(z_coords.min(), z_coords.max())
    ax.set_ylim(y_coords.min(), y_coords.max())
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存图像
    dbg_fig_path = f"{data_dir}/debug_field_slice_{gpu}_freq_{dbg_freq_idx}.png"
    plt.savefig(dbg_fig_path)
    print(f"Debug visualization saved to: {dbg_fig_path}")
    plt.close(fig) # 关闭图像，避免显示出来

    # 清理临时障碍物文件
    # os.remove(dbg_obstacle_path)
    print(f"--- Debugging  Finished ---")

# 自定义sample过程，做指定方向的旋转，不做放缩，不做平移
# 希望能在每次srcidx变化时，scene中的obstacle也能变化
def custom(data_dir, data_name, src_sample_num = None, trg_sample_num = None , 
                             show_scene:bool=False,
                             split_mode:str = 'train',
                             sound_src:str = 'ball.obj',
                             gpu_id = 0
                             ):
    
    
    # 避开Scene对container的正则化
    scene = Scene(f"{data_dir}/config.json", normalize=False)

    # 是否跳过重复数据
    skip = False

    if src_sample_num is None:
        src_sample_num = scene.src_sample_num
    if trg_sample_num is None:
        trg_sample_num = scene.trg_sample_num
        
    for src_idx in tqdm(range(src_sample_num), position=gpu_id+1, desc=f"gpu_{gpu_id}", leave=False):
        random_int = np.random.randint(1000000000)
        seed = random_int % 1000000000
        if skip:
            if os.path.exists(f"{data_dir}/data/{split_mode}_mesh/{data_name}_{src_idx}.obj"):
                # print(f"data exists, gpu_id: {gpu_id} skips.")
                continue
        
        x = torch.zeros(
                trg_sample_num, 3,
                dtype=torch.float32,
            )
        y = torch.zeros(trg_sample_num, 65, dtype=torch.float32)

        # new
        for obj in scene.objs:
            if obj.name != sound_src:
                obj.vertices, obj.triangles = concat_obstacle(data_dir)
                obj.vertices_base = torch.tensor(obj.vertices).cuda().to(torch.float32)
                obj.neumann = torch.zeros(
                    len(obj.vertices_base), dtype=torch.complex64, device="cuda"
                )
                obj.triangles = torch.tensor(obj.triangles, dtype=torch.int32, device='cuda')
                obj.vertices = torch.tensor(obj.vertices, dtype=torch.float32, device='cuda')
        
        if src_idx == 0: # debug
            show_debug(gpu_id, scene, 5)
            show_debug(gpu_id, scene, 10)
            show_debug(gpu_id, scene, 20)
            show_debug(gpu_id, scene, 40)
            mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
            mesh.export(f"{data_dir}/debug_mesh_{gpu_id}.obj")

        for freq_idx in tqdm(range(65), position=gpu_id+1, desc=f"gpu_{gpu_id}, src {src_idx}/{src_sample_num}", leave=False):

            scene.enclose_sample(seed=seed, freq_idx=freq_idx, max_freq_idx=65, sound_source=sound_src,
                            _transition=False, _resize=False, _rotate = False)
            scene.solve()

            x[:, :3] = scene.trg_points
         
            y[:, freq_idx] = scene.potential.abs()#.unsqueeze(-1)
        
        
        if split_mode == 'train':
            torch.save({"x": x, "y": y}, f"{data_dir}/data/train_data/{data_name}_{src_idx}.pt")
            # 以obj格式存储几何形状
            mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
            mesh.export(f"{data_dir}/data/train_mesh/{data_name}_{src_idx}.obj")
        else: # 测试集
            torch.save({"x": x, "y": y}, f"{data_dir}/data/val_data/{data_name}_{src_idx}.pt")
            # 以obj格式存储几何形状
            mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
            mesh.export(f"{data_dir}/data/val_mesh/{data_name}_{src_idx}.obj")


if __name__ == "__main__":

    data_dir = sys.argv[1]
    tag = sys.argv[2]
    gpu_id = int(sys.argv[3])
    mode = 'train'

    src_num = None
    if len(sys.argv) > 4:
        src_num = int(sys.argv[4])
        mode = sys.argv[5]
    
    if mode != 'train':
        src_num = 1 # only 1 source in test and val

    # if os.path.exists(f"{data_dir}/data/{mode}_mesh/out_{tag}_0.obj"):
    #     print(f"data exists, gpu_id: {gpu_id} skips.")
    #     exit()

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print(f"started a process. gpu_id: {gpu_id}")

    custom(data_dir, f"out_{tag}", 
                                  src_sample_num=src_num,
                                  show_scene=False,
                                  split_mode=mode,
                                  sound_src="ball.obj",
                                  gpu_id=gpu_id)

