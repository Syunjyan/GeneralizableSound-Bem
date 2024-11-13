import sys

sys.path.append("./")
import time
import numpy as np
import pybullet_data
import os

import pybullet as p


class ObjMotion:
    def __init__(self, dirname, init_ori, UI=False):
        if UI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.UI = UI
        self.dir = dirname
        filename = self.dir + "/mesh_surf.obj"
        # p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_idx = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        # useMaximalCoordinates is much faster then the default reduced coordinates (Featherstone)
        # disable rendering during creation.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # disable tinyrenderer, software (CPU) renderer, we don't use it here
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        shift = [0, 0, 0]
        meshScale = [1.0, 1.0, 1.0]
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=filename,
            collisionFramePosition=shift,
            meshScale=meshScale,
        )
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=filename,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=shift,
            meshScale=meshScale,
        )
        self.obj_id = p.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0, 0, 2.0],
            baseOrientation=init_ori,
            useMaximalCoordinates=True,
        )
        p.setGravity(0, 0, -10)
        p.resetBaseVelocity(self.obj_id, linearVelocity=[0, 0, 0])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setRealTimeSimulation(1)
        # pybullet elasticity
        p.changeDynamics(self.obj_id, -1, lateralFriction=0.2, restitution=0.1)
        p.changeDynamics(self.plane_idx, -1, lateralFriction=0.2, restitution=0.01)

    def run(self, time_all=1, time_step=1 / 2000, save_data=False):
        p.setTimeStep(time_step)
        time_step_num = int(time_all / time_step)
        motion_data = np.zeros((time_step_num, 7))
        contact_data = np.zeros((time_step_num, 32, 7))
        for i in range(time_step_num):
            p.stepSimulation()
            Pos, Ori = p.getBasePositionAndOrientation(self.obj_id)
            motion_data[i, :3] = Pos
            motion_data[i, 3:] = Ori
            if self.UI:
                time.sleep(time_step)
            points_data = p.getContactPoints(bodyB=self.obj_id)
            idx = 0
            for data in points_data:
                contact_data[i, idx, :3] = data[6]
                contact_data[i, idx, 3:6] = data[7]
                contact_data[i, idx, 6] = data[9]
                idx += 1
                if idx >= 32:
                    break
        if save_data:
            np.savez_compressed(
                self.dir + "/contact",
                pos=contact_data[..., :3],
                normal=contact_data[..., 3:6],
                force=contact_data[..., 6],
            )
            np.savez_compressed(
                self.dir + "/motion",
                pos=motion_data[:, :3],
                ori=motion_data[:, 3:],
                step=time_step,
            )
        return self

    def close(self):
        p.disconnect()


obj_dir = "dataset/modal_vibration/"

from glob import glob

sub_dirs = glob(obj_dir + "/*")
for sub_dir in sub_dirs:
    if os.path.isdir(sub_dir):
        print(sub_dir)
        objMotion = (
            ObjMotion(sub_dir, [-0.5, -0.2, -0.1, 0.7], UI=False)
            .run(time_all=3, time_step=1 / 20000, save_data=True)
            .close()
        )
