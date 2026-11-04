"""Diagnose full-Pro BEM spikes without modifying generated samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "experiments" / "paper_next" / "pro_relative_full_manifest.json"
DEFAULT_DATASET = Path(
    "/data5/houxinyun/GeneralizableSound/data/pro_relative_full.staging"
)


def corrected_bicgstab(torch, lhs, rhs, max_steps: int, tolerance: float = 1e-5):
    """BiCGSTAB with Hermitian products for a complex-valued system."""
    x = torch.zeros_like(rhs)
    r = rhs - lhs @ x
    r_hat = r.clone()
    rho_old = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    alpha = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    omega = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    v = torch.zeros_like(rhs)
    p = torch.zeros_like(rhs)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(1e-30)

    for iteration in range(1, max_steps + 1):
        rho = torch.vdot(r_hat, r)
        if rho.abs() < 1e-30:
            break
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = lhs @ p
        denominator = torch.vdot(r_hat, v)
        if denominator.abs() < 1e-30:
            break
        alpha = rho / denominator
        s = r - alpha * v
        if torch.linalg.vector_norm(s) / rhs_norm < tolerance:
            x = x + alpha * p
            return x, iteration, True
        t = lhs @ s
        tt = torch.vdot(t, t)
        if tt.abs() < 1e-30:
            break
        omega = torch.vdot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        if torch.linalg.vector_norm(r) / rhs_norm < tolerance:
            return x, iteration, True
        if omega.abs() < 1e-30:
            break
        rho_old = rho
    return x, iteration, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="bem_enclosed_obj028_sample008")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=(4115, 4120, 4124, 4125, 4126, 4130, 4135),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import numpy as np
    import torch
    import trimesh

    from experiments.paper_next.generate_pro_full import prepare_bem_config
    from src.bem.BiCGSTAB import BiCGSTAB
    from src.bem.solver import BEM_Solver, solve_linear_equation_direct
    from src.scene import Scene

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    spec = next(item for item in manifest["scenes"] if item["stem"] == args.stem)
    if spec["solver"] != "bem":
        raise ValueError("The diagnostic only supports BEM scenes.")

    work_dir = args.dataset / "_diagnostics" / args.stem
    config_path = prepare_bem_config(
        work_dir, ROOT / spec["source"], spec["query_count"]
    )
    scene = Scene(str(config_path))
    target_frequency = args.frequencies[len(args.frequencies) // 2]
    base_index = int(round(target_frequency / 125.0))
    if spec["category"] == "general":
        scene.my_sample(
            seed=spec["seed"],
            freq_idx=base_index,
            max_freq_idx=65,
            sound_source="ball.obj",
        )
    else:
        scene.enclose_sample(
            seed=spec["seed"],
            freq_idx=base_index,
            max_freq_idx=65,
            sound_source="ball.obj",
        )

    # This reproduces the original target-point RNG state and records the
    # production solver result at the central frequency.
    scene.solve()
    target_points = scene.trg_points.clone()
    production_potential = scene.potential.clone()
    source = next(obj for obj in scene.objs if obj.name == "ball.obj")
    center = (
        source.vertices.max(dim=0).values + source.vertices.min(dim=0).values
    ) / 2.0
    radius = (
        source.vertices.max(dim=0).values - source.vertices.min(dim=0).values
    ).mean() / 2.0
    distances = torch.linalg.vector_norm(target_points - center, dim=-1).cpu()
    neumann_magnitude = source.neumann[0].abs().cpu()

    saved_path = (
        args.dataset / f"{spec['split']}_data" / f"{spec['stem']}.pt"
    )
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    saved_mesh_path = (
        args.dataset / f"{spec['split']}_mesh" / f"{spec['stem']}.obj"
    )
    saved_mesh = trimesh.load_mesh(saved_mesh_path, process=False)
    reproduced_mesh = scene.get_obstacle_mesh()
    production_denominator = (
        neumann_magnitude
        * radius.cpu().square()
        / (
            distances
            * torch.sqrt(
                1.0
                + torch.as_tensor(
                    scene.k * radius.item(), dtype=distances.dtype
                ).square()
            )
        )
    ).clamp_min(1e-12)
    production_transfer = production_potential.abs() / production_denominator
    results = []
    for frequency in args.frequencies:
        k = 2.0 * np.pi * frequency / 343.2
        solver = BEM_Solver(scene.vertices, scene.triangles)
        identity = solver.identity_matrix()
        beta = 1j / k
        lhs = (
            solver.assemble_boundary_matrix(k, "bm_lhs", approx=True)
            + 0.5 * identity
        )
        rhs = (
            solver.assemble_boundary_matrix(k, "bm_rhs", approx=True)
            - beta * 0.5 * identity
        ) @ scene.neumann

        step_counts = (
            (500, 500, 500, 2000)
            if frequency == target_frequency
            else (500,)
        )
        for repeat, steps in enumerate(step_counts):
            linear_solver = BiCGSTAB(lambda value: lhs @ value)
            dirichlet = linear_solver.solve(
                rhs, nsteps=steps, tol=1e-10, atol=1e-16
            )
            residual = rhs - lhs @ dirichlet
            relative_residual = float(
                torch.linalg.vector_norm(residual)
                / torch.linalg.vector_norm(rhs).clamp_min(1e-30)
            )
            potential = solver.boundary2potential(
                k, scene.neumann, dirichlet, target_points
            ).abs().cpu()
            denominator = (
                neumann_magnitude
                * radius.cpu().square()
                / (
                    distances
                    * torch.sqrt(
                        1.0
                        + torch.as_tensor(
                            k * radius.item(), dtype=distances.dtype
                        ).square()
                    )
                )
            ).clamp_min(1e-12)
            transfer = potential / denominator
            results.append(
                {
                    "frequency_hz": frequency,
                    "repeat": repeat,
                    "max_steps": steps,
                    "relative_residual": relative_residual,
                    "dirichlet_l2": float(
                        torch.linalg.vector_norm(dirichlet).cpu()
                    ),
                    "potential_max": float(potential.max()),
                    "transfer_min": float(transfer.min()),
                    "transfer_median": float(transfer.median()),
                    "transfer_max": float(transfer.max()),
                    "all_finite": bool(torch.isfinite(transfer).all()),
                }
            )
        if frequency == target_frequency:
            for steps in (500, 2000):
                dirichlet, iterations, converged = corrected_bicgstab(
                    torch, lhs, rhs, steps
                )
                residual = rhs - lhs @ dirichlet
                relative_residual = float(
                    torch.linalg.vector_norm(residual)
                    / torch.linalg.vector_norm(rhs).clamp_min(1e-30)
                )
                potential = solver.boundary2potential(
                    k, scene.neumann, dirichlet, target_points
                ).abs().cpu()
                denominator = (
                    neumann_magnitude
                    * radius.cpu().square()
                    / (
                        distances
                        * torch.sqrt(
                            1.0
                            + torch.as_tensor(
                                k * radius.item(), dtype=distances.dtype
                            ).square()
                        )
                    )
                ).clamp_min(1e-12)
                transfer = potential / denominator
                results.append(
                    {
                        "frequency_hz": frequency,
                        "solver_variant": "hermitian_bicgstab",
                        "max_steps": steps,
                        "iterations": iterations,
                        "converged": converged,
                        "relative_residual": relative_residual,
                        "dirichlet_l2": float(
                            torch.linalg.vector_norm(dirichlet).cpu()
                        ),
                        "potential_max": float(potential.max()),
                        "transfer_min": float(transfer.min()),
                        "transfer_median": float(transfer.median()),
                        "transfer_max": float(transfer.max()),
                        "all_finite": bool(torch.isfinite(transfer).all()),
                    }
                )
            dirichlet, solve_info = solve_linear_equation_direct(
                lhs,
                rhs,
                residual_tolerance=1e-6,
                refinement_steps=1,
            )
            potential = solver.boundary2potential(
                k, scene.neumann, dirichlet, target_points
            ).abs().cpu()
            denominator = (
                neumann_magnitude
                * radius.cpu().square()
                / (
                    distances
                    * torch.sqrt(
                        1.0
                        + torch.as_tensor(
                            k * radius.item(), dtype=distances.dtype
                        ).square()
                    )
                )
            ).clamp_min(1e-12)
            transfer = potential / denominator
            results.append(
                {
                    "frequency_hz": frequency,
                    "solver_variant": "direct_lu_refined",
                    **solve_info,
                    "dirichlet_l2": float(
                        torch.linalg.vector_norm(dirichlet).cpu()
                    ),
                    "potential_max": float(potential.max()),
                    "transfer_min": float(transfer.min()),
                    "transfer_median": float(transfer.median()),
                    "transfer_max": float(transfer.max()),
                    "all_finite": bool(torch.isfinite(transfer).all()),
                }
            )
        del lhs, rhs, identity, solver
        torch.cuda.empty_cache()

    report = {
        "scene": spec,
        "reproduction": {
            "listener_max_abs_difference": float(
                (saved["x"] - target_points.cpu()).abs().max()
            ),
            "mesh_vertex_count_saved": int(len(saved_mesh.vertices)),
            "mesh_vertex_count_reproduced": int(len(reproduced_mesh.vertices)),
            "mesh_face_count_saved": int(len(saved_mesh.faces)),
            "mesh_face_count_reproduced": int(len(reproduced_mesh.faces)),
            "mesh_vertex_max_abs_difference": (
                float(
                    np.max(
                        np.abs(saved_mesh.vertices - reproduced_mesh.vertices)
                    )
                )
                if saved_mesh.vertices.shape == reproduced_mesh.vertices.shape
                else None
            ),
            "mesh_faces_identical": bool(
                saved_mesh.faces.shape == reproduced_mesh.faces.shape
                and np.array_equal(saved_mesh.faces, reproduced_mesh.faces)
            ),
        },
        "saved_target_bin": {
            "frequency_index": base_index,
            "frequency_hz": target_frequency,
            "transfer_min": float(saved["y"][:, base_index].min()),
            "transfer_median": float(saved["y"][:, base_index].median()),
            "transfer_max": float(saved["y"][:, base_index].max()),
        },
        "production_solve": {
            "potential_max": float(production_potential.abs().max()),
            "transfer_min": float(production_transfer.min()),
            "transfer_median": float(production_transfer.median()),
            "transfer_max": float(production_transfer.max()),
        },
        "results": results,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
