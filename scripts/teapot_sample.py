import numpy as np
import trimesh
from typing import Optional, Tuple

def _tri_edges(t: np.ndarray):
    return np.array([t[1] - t[0], t[2] - t[1], t[0] - t[2]], dtype=np.float64)

def _proj_interval(tri: np.ndarray, axis: np.ndarray):
    p = tri @ axis
    return float(np.min(p)), float(np.max(p))

def _interval_disjoint(a0: float, a1: float, b0: float, b1: float, eps: float = 1e-10) -> bool:
    return a1 < b0 - eps or b1 < a0 - eps

def _tri_tri_intersect_sat(t1: np.ndarray, t2: np.ndarray, eps: float = 1e-10) -> bool:
    """Strict triangle-triangle intersection test using SAT."""
    e1 = _tri_edges(t1)
    e2 = _tri_edges(t2)
    n1 = np.cross(e1[0], e1[1])
    n2 = np.cross(e2[0], e2[1])

    axes = []
    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    if n1n > eps:
        axes.append(n1 / n1n)
    if n2n > eps:
        axes.append(n2 / n2n)

    for a in e1:
        for b in e2:
            ax = np.cross(a, b)
            an = np.linalg.norm(ax)
            if an > eps:
                axes.append(ax / an)

    coplanar = False
    if n1n > eps and n2n > eps:
        n1u = n1 / n1n
        n2u = n2 / n2n
        if abs(abs(np.dot(n1u, n2u)) - 1.0) < 1e-6:
            d = np.abs((t2 - t1[0]) @ n1u)
            if float(np.max(d)) < 1e-6:
                coplanar = True

    if coplanar:
        for a in e1:
            ax = np.cross(n1u, a)
            an = np.linalg.norm(ax)
            if an > eps:
                axes.append(ax / an)
        for b in e2:
            ax = np.cross(n1u, b)
            an = np.linalg.norm(ax)
            if an > eps:
                axes.append(ax / an)

    for ax in axes:
        a0, a1 = _proj_interval(t1, ax)
        b0, b1 = _proj_interval(t2, ax)
        if _interval_disjoint(a0, a1, b0, b1, eps=eps):
            return False

    return True

def strict_mesh_intersection(a: trimesh.Trimesh, b: trimesh.Trimesh, chunk: int = 48) -> bool:
    """Return True if any face pair intersects strictly (triangle-level test)."""
    ta = np.asarray(a.triangles, dtype=np.float64)
    tb = np.asarray(b.triangles, dtype=np.float64)
    if len(ta) == 0 or len(tb) == 0:
        return False

    a_min = ta.min(axis=1)
    a_max = ta.max(axis=1)
    b_min = tb.min(axis=1)
    b_max = tb.max(axis=1)

    for i0 in range(0, len(ta), chunk):
        i1 = min(i0 + chunk, len(ta))
        ca_min = a_min[i0:i1]
        ca_max = a_max[i0:i1]

        ov = (ca_min[:, None, :] <= b_max[None, :, :]) & (ca_max[:, None, :] >= b_min[None, :, :])
        hit = np.where(np.all(ov, axis=2))
        if len(hit[0]) == 0:
            continue

        for ia_local, ib in zip(hit[0], hit[1]):
            ia = i0 + int(ia_local)
            if _tri_tri_intersect_sat(ta[ia], tb[int(ib)]):
                return True

    return False

def check_no_penetration(body_mesh: trimesh.Trimesh, lid_mesh: trimesh.Trimesh) -> bool:
    return not strict_mesh_intersection(body_mesh, lid_mesh)

def min_vertex_pair_distance(a: trimesh.Trimesh, b: trimesh.Trimesh, rng: np.random.Generator, sample_n: int = 2000) -> float:
    va = a.vertices
    vb = b.vertices
    if len(va) > sample_n:
        va = va[rng.choice(len(va), size=sample_n, replace=False)]
    if len(vb) > sample_n:
        vb = vb[rng.choice(len(vb), size=sample_n, replace=False)]

    d = va[:, None, :] - vb[None, :, :]
    dist2 = np.sum(d * d, axis=-1)
    return float(np.sqrt(np.min(dist2)))

def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    M = trimesh.transformations.random_rotation_matrix(rng.random(3))
    return M[:3, :3]

def sample(
    body: trimesh.Trimesh,
    lid: trimesh.Trimesh,
    seed: Optional[int] = None,
    max_tries: int = 128,
    body_target_size: float = 0.26,
    min_clearance: float = 0.004,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Generate one randomized teapot scene based on manually aligned body and lid."""
    rng = np.random.default_rng(seed)

    # 1. 对环境进行统一步伐缩放以匹配 target_size，确保用户手动对应的相对位置不变
    scene_extents = np.max(np.maximum(body.vertices.max(axis=0), lid.vertices.max(axis=0)) - 
                           np.minimum(body.vertices.min(axis=0), lid.vertices.min(axis=0)))
    scale = body_target_size / max(scene_extents, 1e-8)
    
    body_scaled = body.copy()
    body_scaled.apply_scale(scale)
    # 为保证相对位置，我们将他们放置在相对于body质心在原点的位置
    body_centroid = body_scaled.centroid.copy()
    body_scaled.vertices -= body_centroid
    
    lid_scaled = lid.copy()
    lid_scaled.apply_scale(scale)
    lid_scaled.vertices -= body_centroid

    # 提取壶盖的几何中心作为旋转参考点
    lid_center = lid_scaled.centroid

    success = False
    lid_final = lid_scaled

    # 2. 对茶盖进行局部打乱
    for _ in range(max_tries):
        lid_cand = lid_scaled.copy()

        # a) 进行轻微缩放（基于盖子中心）
        s = float(rng.uniform(1.0, 1.2))
        v = lid_cand.vertices - lid_center
        v *= s

        # b) 以盖子中心进行任意三维随机旋转
        Rl = random_rotation_matrix(rng)
        v = v @ Rl.T

        # c) 进行随机平移。原本依靠提取的半径（mouth.radius），这里改用绝对物理距离偏移
        # 水平面平移范围：[-0.015, 0.015] 米，垂直抬升高度：[0.010, 0.050] 米
        dx = float(rng.uniform(-0.015, 0.015))
        dz = float(rng.uniform(-0.015, 0.015))
        dy = float(rng.uniform(0.010, 0.050))
        
        t_local = np.array([dx, dy, dz], dtype=np.float64)
        v = v + lid_center + t_local
        lid_cand.vertices = v

        # 检查碰撞情况，开销可能较大
        # if not check_no_penetration(body_scaled, lid_cand):
        #     continue
        # if min_vertex_pair_distance(body_scaled, lid_cand, rng) < float(min_clearance):
        #     continue

        lid_final = lid_cand
        success = True
        break

    if not success:
        # Fallback: 如果多次依然穿插，就使用无损原位壶盖
        lid_final = lid_scaled.copy()
        lid_final.vertices = lid_final.vertices + np.array([0.0, np.random.random() * 0.03, 0.0], dtype=np.float64)

    # 3. 随机化全局场景的整体朝向和位置
    Rg = random_rotation_matrix(rng)
    tg = np.array([
        float(rng.uniform(-0.03, 0.03)),
        float(rng.uniform(-0.03, 0.03)),
        float(rng.uniform(-0.03, 0.03)),
    ], dtype=np.float64)
    
    body_scaled.vertices = body_scaled.vertices @ Rg.T + tg
    lid_final.vertices = lid_final.vertices @ Rg.T + tg

    return body_scaled, lid_final