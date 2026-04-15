"""
Two-object bin-picking environment for Meta-World v3 (Farama).

Drop-in replacement for SawyerBinPickingEnvV3.
Place this file at:
    <metaworld_install>/envs/sawyer_bin_picking_v3_two_objects.py

Then in metaworld/envs/__init__.py add:
    from metaworld.envs.sawyer_bin_picking_v3_two_objects import SawyerBinPickingTwoObjEnvV3
"""

from __future__ import annotations

from typing import Any
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class SawyerBinPickingTwoObjEnvV3(SawyerXYZEnv):
    """
    Bin-picking with two independently spawned objects.

    The robot must pick EITHER object and drop it in the target bin.
    Reward = max(reward_for_obj1, reward_for_obj2), so the agent gets
    full credit for whichever object it decides to work on.

    Observation layout  (39 dims total):
        obs[0:3]   – EE xyz
        obs[3]     – gripper openness
        obs[4:7]   – obj1 xyz        (same slot as original single-obj env)
        obs[7:11]  – obj1 quaternion
        obs[11:14] – obj2 xyz        ← NEW
        obs[14:18] – obj2 quaternion ← NEW
        obs[18:25] – goal xyz (repeated / padded by SawyerXYZEnv)
    """

    _MIN_OBJ_DIST: float = 0.07   # minimum xy separation at spawn

    @classmethod
    def make(cls, **kwargs) -> "SawyerBinPickingTwoObjEnvV3":
        """Convenience constructor that handles set_task boilerplate.

        Usage:
            env = SawyerBinPickingTwoObjEnvV3.make()
            obs, info = env.reset()
        """
        import pickle
        from metaworld.types import Task
        env = cls(**kwargs)
        task = Task(
            env_name="bin-picking-two-obj-v3",
            data=pickle.dumps({
                "env_cls": cls,
                "rand_vec": env._random_reset_space.sample(),
                "partially_observable": False,
            }),
        )
        env.set_task(task)
        return env

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low  = (-0.5, 0.40, 0.07)
        hand_high = ( 0.5, 1.00, 0.50)
        obj_low   = (-0.21, 0.65, 0.02)
        obj_high  = (-0.03, 0.75, 0.02)
        goal_low  = np.array([0.1199, 0.699, -0.001])
        goal_high = np.array([0.1201, 0.701,  0.001])

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        self.reward_function_version = reward_function_version

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos":   np.array([-0.12, 0.70, 0.02]),
            "hand_init_pos":  np.array([ 0.00, 0.60, 0.20]),
        }
        self.goal          = np.array([0.12, 0.7, 0.02])
        self.obj_init_pos  = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        # second object – will be overwritten each reset
        self.obj2_init_pos = np.array([-0.05, 0.70, 0.02])

        self.liftThresh = 0.1
        self._target_to_obj_init:  float | None = None
        self._target_to_obj2_init: float | None = None

        self.goal_space = Box(goal_low, goal_high, dtype=np.float64)
        self._random_reset_space = Box(
            np.hstack((obj_low,  goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    @property
    def model_name(self) -> str:
        # Reuse the existing bin-picking XML; obj2 body is added via
        # _get_pos_objects / _get_quat_objects and set via set_state.
        # We rely on the XML already having an "obj2" body – see note below.
        return full_V3_path_for("sawyer_xyz/sawyer_bin_picking_two_obj.xml")

    # ------------------------------------------------------------------ #
    #  Object accessors (called by SawyerXYZEnv._get_obs)                 #
    # ------------------------------------------------------------------ #
    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("objGeom")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Return concatenated positions of both objects (6,)."""
        return np.concatenate([
            self.get_body_com("obj"),
            self.get_body_com("obj2"),
        ])

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        """Return concatenated quaternions of both objects (8,)."""
        return np.concatenate([
            self.data.body("obj").xquat,
            self.data.body("obj2").xquat,
        ])

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #
    def _sample_obj2_pos(self, obj1_pos: np.ndarray) -> np.ndarray:
        """Sample a spawn for obj2 far enough from obj1."""
        low  = np.array([-0.21, 0.65, 0.02])
        high = np.array([-0.03, 0.75, 0.02])
        for _ in range(200):
            pos = self.np_random.uniform(low, high)
            if np.linalg.norm(pos[:2] - obj1_pos[:2]) >= self._MIN_OBJ_DIST:
                return pos
        # fallback: mirror obj1 in x
        return np.array([-obj1_pos[0] * 0.5, obj1_pos[1], obj1_pos[2]])

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self._target_pos    = self.goal.copy()
        self.obj_init_pos   = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]

        # ---- place obj1 (same logic as original) ----
        obj_height = self.get_body_com("obj")[2]
        rand_vec = self._get_state_rand_vec()
        xy = rand_vec[:2]
        self.obj_init_pos = np.concatenate([xy, [obj_height]])
        self._set_obj_xyz(self.obj_init_pos)

        # ---- place obj2 ----
        self.obj2_init_pos = self._sample_obj2_pos(self.obj_init_pos)
        self._set_obj2_xyz(self.obj2_init_pos)

        # ---- shared bookkeeping ----
        self._target_pos          = self.get_body_com("bin_goal")
        self._target_to_obj_init  = None
        self._target_to_obj2_init = None

        self.objHeight    = self.data.body("obj").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh

        self.maxPlacingDist = (
            np.linalg.norm(self.obj_init_pos[:2] - self._target_pos[:2])
            + self.heightTarget
        )
        self.maxPlacingDist2 = (
            np.linalg.norm(self.obj2_init_pos[:2] - self._target_pos[:2])
            + self.heightTarget
        )

        self.placeCompleted = False
        self.pickCompleted  = False

        return self._get_obs()



    def _set_obj2_xyz(self, pos: np.ndarray) -> None:
        """Teleport obj2 to pos with zero velocity."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        jnt  = self.model.joint("obj2_joint")
        addr = self.model.jnt_qposadr[jnt.id]
        qpos[addr:addr + 3]  = pos
        qpos[addr + 3:addr + 7] = [1, 0, 0, 0]   # identity quaternion
        dof_addr = self.model.jnt_dofadr[jnt.id]
        qvel[dof_addr:dof_addr + 6] = 0.0
        self.set_state(qpos, qvel)

    # ------------------------------------------------------------------ #
    #  Reward                                                              #
    # ------------------------------------------------------------------ #
    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            near_object,
            grasp_success,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        info = {
            "success":        float(obj_to_target <= 0.05),
            "near_object":    float(near_object),
            "grasp_success":  float(grasp_success),
            "grasp_reward":   grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target":  obj_to_target,
            "unscaled_reward": reward,
        }
        return reward, info

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[Any]
    ) -> tuple[float, bool, bool, float, float, float]:
        assert self.obj_init_pos is not None and self._target_pos is not None

        hand = obs[:3]
        obj1 = obs[4:7]
        # obs layout from _get_pos_objects / _get_quat_objects:
        # [ee(4), obj1_pos(3), obj1_quat(4), obj2_pos(3), obj2_quat(4), goal...]
        obj2 = obs[11:14]

        def _single_obj_reward(obj, obj_init_pos, target_to_obj_init_ref):
            target_to_obj = float(np.linalg.norm(obj - self._target_pos))
            if target_to_obj_init_ref[0] is None:
                target_to_obj_init_ref[0] = target_to_obj

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=target_to_obj_init_ref[0],
                sigmoid="long_tail",
            )

            threshold = 0.03
            radii = [
                np.linalg.norm(hand[:2] - obj_init_pos[:2]),
                np.linalg.norm(hand[:2] - self._target_pos[:2]),
            ]
            floor = min(
                0.02 * np.log(r - threshold) + 0.2 if r > threshold else 0.0
                for r in radii
            )
            above_floor = (
                1.0 if hand[2] >= floor
                else reward_utils.tolerance(
                    max(floor - hand[2], 0.0),
                    bounds=(0.0, 0.01),
                    margin=0.05,
                    sigmoid="long_tail",
                )
            )

            object_grasped = self._gripper_caging_reward(
                action, obj,
                obj_radius=0.015,
                pad_success_thresh=0.05,
                object_reach_radius=0.01,
                xz_thresh=0.01,
                desired_gripper_effort=0.7,
                high_density=True,
            )
            reward = reward_utils.hamacher_product(object_grasped, in_place)

            near   = bool(np.linalg.norm(obj - hand) < 0.04)
            pinched = bool(obs[3] < 0.43)
            lifted  = bool(obj[2] - 0.02 > obj_init_pos[2])
            grasped = near and lifted and not pinched

            if grasped:
                reward += 1.0 + 5.0 * reward_utils.hamacher_product(above_floor, in_place)
            if target_to_obj < self.TARGET_RADIUS:
                reward = 10.0

            return reward, near, grasped, target_to_obj, object_grasped, in_place

        # mutable boxes so the helper can update the cached init distance
        ref1 = [self._target_to_obj_init]
        ref2 = [self._target_to_obj2_init]

        r1, near1, grasp1, d1, og1, ip1 = _single_obj_reward(obj1, self.obj_init_pos,  ref1)
        r2, near2, grasp2, d2, og2, ip2 = _single_obj_reward(obj2, self.obj2_init_pos, ref2)

        # persist cached init distances
        self._target_to_obj_init  = ref1[0]
        self._target_to_obj2_init = ref2[0]

        # take whichever object gives higher reward
        if r1 >= r2:
            return r1, near1, grasp1, d1, og1, ip1
        else:
            return r2, near2, grasp2, d2, og2, ip2