import sys

sys.path.append("./")

import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
from src.mcs.mcs import ImportanceSampler, MonteCarloWeight
from src.timer import Timer
from src.solver import BiCGSTAB, BiCGSTAB_batch
from numba import njit

data_dir = "dataset/ABC/data/compare"
data_list = glob(os.path.join(data_dir, "*.npz"))[:100]


def save_ffat_maps(ffat_map, img_path):
    if isinstance(ffat_map, torch.Tensor):
        ffat_map = ffat_map.cpu().numpy()

    rows = 4
    cols = ffat_map.shape[0] // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i in range(rows):
        for j in range(cols):
            img = axes[i, j].imshow(ffat_map[i * cols + j])
            fig.colorbar(img, ax=axes[i, j])  # Add color bar for each subplot

    plt.savefig(img_path)
    plt.close()


mode_num = 16
bem_err = np.zeros((len(data_list), mode_num))
ours_err = np.zeros((len(data_list), mode_num))
neuralsound_err = np.zeros((len(data_list), mode_num))
bem_time = []
ours_time = []
neuralsound_time = []

for i, data_path in enumerate(tqdm(data_list)):
    print(data_path)
    eigen_path = data_path.replace("compare", "eigen")
    data = np.load(eigen_path)
    vertices, triangles = data["vertices"], data["triangles"]
    neumann_bem, dirichlet_bem = data["neumann"], data["dirichlet"]

    wave_number = data["wave_number"]
    bem_cost_time = data["bem_cost_time"]
    mode_num = len(wave_number)
    vertices = torch.from_numpy(vertices).cuda().to(torch.float32)
    triangles = torch.from_numpy(triangles).cuda().to(torch.int32)
    ks = torch.from_numpy(wave_number).cuda().to(torch.float32)
    neumann = torch.from_numpy(neumann_bem).cuda().to(torch.complex64).T.unsqueeze(-1)
    # print(neumann.shape)  # (batch_size, n, 1)

    timer = Timer(log_output=True)
    importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
    sampler = ImportanceSampler(vertices, triangles, importance, 100000)
    sampler.update()
    sampler.poisson_disk_resample(0.004, 4)
    timer.log("sample points: ", sampler.num_samples, record=True)

    G0_constructor = MonteCarloWeight(sampler.points, sampler)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
    G0_batch = G0_constructor.get_weights_boundary_ks(ks)
    G1_batch = G1_constructor.get_weights_boundary_ks(ks)
    neumann = neumann[:, sampler.points_index, :]
    b_batch = torch.bmm(G0_batch, neumann).permute(1, 2, 0)
    timer.log("construct G and b", record=True)

    solver = BiCGSTAB_batch(
        lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
    )
    timer.log("construct A", record=True)

    dirichlet = solver.solve(b_batch, tol=1e-6, nsteps=200).permute(2, 0, 1)
    timer.log("solve", record=True)

    cost_time = timer.record_time

    image_size = 32
    length = 0.15
    points_ = unit_sphere_surface_points(image_size) * length
    points_list = [points_ * 1.25, points_ * 1.25**2, points_ * 1.25**3]

    ffat_maps_ours = []
    for points in points_list:
        points = torch.from_numpy(points).cuda().float().reshape(-1, 3)
        points_batch = 256
        idx = 0
        ffat_map = torch.zeros((mode_num, image_size * 2 * image_size)).cuda()
        while idx < len(points):
            stride = min(points_batch, len(points) - idx)
            sub_points = points[idx : idx + stride]
            G0_constructor = MonteCarloWeight(sub_points, sampler)
            G1_constructor = MonteCarloWeight(sub_points, sampler, deriv=True)
            G0 = G0_constructor.get_weights_potential_ks(ks)
            G1 = G1_constructor.get_weights_potential_ks(ks)
            # print(G0.shape, G1.shape, dirichlet.shape, neumann.shape)
            RHS = G1 @ dirichlet - G0 @ neumann
            ffat_map[:, idx : idx + stride] = RHS.reshape(mode_num, -1).abs()
            idx += stride
        ffat_map = ffat_map.reshape(mode_num, image_size * 2, image_size)
        ffat_maps_ours.append(ffat_map.cpu().numpy())
        # print(ffat_map.shape)

    data = np.load(data_path)
    bem, neuralsound = data["bem"], data["neuralsound"]
    ours = np.array(ffat_maps_ours)
    bem_time_, neuralsound_time_ = data["bem_time"], data["neuralsound_time"]
    ours_time_ = cost_time

    bem = bem.reshape(mode_num, -1)
    neuralsound = neuralsound.reshape(mode_num, -1)
    ours = ours.reshape(mode_num, -1)

    err1 = ((ours - bem) ** 2).mean(axis=1) / ((bem) ** 2).mean(axis=1)
    err2 = ((neuralsound - bem) ** 2).mean(axis=1) / ((bem) ** 2).mean(axis=1)
    ours_err[i] = err1
    neuralsound_err[i] = err2
    if err1.mean() > err2.mean():
        print(data_path, ":", err1.mean(), err2.mean())
        # for i in range(3):
        #     img_path = data_path.replace(".npz", f"_ours_{i}.png").replace(
        #         "compare", "fail"
        #     )
        #     os.makedirs(os.path.dirname(img_path), exist_ok=True)
        #     save_ffat_maps(ours[i], img_path)
        #     save_ffat_maps(bem[i], img_path.replace("ours", "bem"))
        #     save_ffat_maps(neuralsound[i], img_path.replace("ours", "neuralsound"))

    bem_time.append(bem_time_)
    ours_time.append(ours_time_)
    neuralsound_time.append(neuralsound_time_)

bem_err = bem_err[1:].mean(-1)
ours_err = ours_err[1:].mean(-1)
neuralsound_err = neuralsound_err[1:].mean(-1)

print(len(ours_err[torch.isnan(ours_err)]))
# analysis and plot
print("bem_err", bem_err.mean(), "+-", bem_err.std())
print("ours_err", ours_err.mean(), "+-", ours_err.std())
print("neuralsound_err", neuralsound_err.mean(), "+-", neuralsound_err.std())

plt.plot(ours_err, label="ours")
plt.plot(neuralsound_err, label="neuralsound")
plt.legend()
plt.savefig("ours_neuralsound_err.png")
plt.clf()
plt.close()

bem_time = np.array(bem_time)[1:]
ours_time = np.array(ours_time)[1:]
neuralsound_time = np.array(neuralsound_time)[1:]

print("bem_time", bem_time.mean(), "+-", bem_time.std())
print("ours_time", ours_time.mean(), "+-", ours_time.std())
print("neuralsound_time", neuralsound_time.mean(), "+-", neuralsound_time.std())

plt.plot(ours_time, label="ours")
plt.plot(neuralsound_time, label="neuralsound")
plt.legend()
plt.savefig("ours_neuralsound_time.png")
plt.clf()
plt.close()
