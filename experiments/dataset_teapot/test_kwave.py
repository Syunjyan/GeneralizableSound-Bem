import sys
import os
import numpy as np
import scipy.io.wavfile as wavfile
import trimesh
import matplotlib.pyplot as plt

# Add the project root to sys.path so we can import 'core' and 'utils'
sys.path.append("./")

from kwave_solver import KWaveSolver

data_dir = "dataset/teapot"
grid_size = (128, 128, 128)
resolution = 3e-3

Debug = False

def main():
    """
    Reproduce ir2mask.ipynb:
    1. Load Teapot + Lid (Combined)
    2. Solve for single IR with obstacle
    3. Solve for single IR without obstacle (free field)
    4. Convolve with music
    5. Calculate frequency mask (BEM Mask)
    """
    
    # 1. Initialize Solver
    solver = KWaveSolver(resolution=0.005, time_end=0.03)
    
    # 2. Add Objects
    try:
        # Load Teapot and Lid
        teapot_path = os.path.join(data_dir, 'obstacles/00018/0.obj')
        
        mesh_teapot = trimesh.load(teapot_path)
        mesh_teapot.apply_scale(0.26/np.max(mesh_teapot.extents))
        
        # Combine them as in notebook: mesh = mesh + gaizi
        combined_mesh = mesh_teapot
        
        solver.add_object(combined_mesh)
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    

    # 3. Set Target (Sensor)
    trg_point = np.array([0.4, 0.4, 0.4])

    
    # 4. Solve with obstacle
    print("Running Simulation (With Obstacle)...")
    spec = solver.solve_spec(trg_point)[0]



if __name__ == "__main__":
    main()
