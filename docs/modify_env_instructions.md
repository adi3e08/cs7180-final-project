# Two-Object Bin Picking Environment — Change Summary

## Goal
Extend Meta-World's `bin-picking-v3` environment to include a second, independently
spawned object, making it more visually interesting and a richer benchmark for
pick-and-place imitation learning / RL agents.

---

## New Files

### `metaworld/envs/sawyer_bin_picking_v3_two_objects.py`
A new environment class `SawyerBinPickingTwoObjEnvV3` that subclasses `SawyerBinPickingEnvV3`.

Key changes over the parent:
- **`_get_pos_objects` / `_get_quat_objects`** — return concatenated positions and
  quaternions for both `obj` and `obj2`, which `SawyerXYZEnv._get_obs()` uses to
  build the flat observation automatically.
- **Observation space** — extended by 7 dims (xyz + quaternion for obj2), giving a
  total of 39 dims vs. the original's 25.
- **`reset_model`** — places obj1 using the standard random vec, then samples an
  independent position for obj2 with a minimum xy separation of 7 cm
  (`_MIN_OBJ_DIST`) to avoid spawning them on top of each other.
- **`_set_obj2_xyz`** — teleports obj2 by writing directly into MuJoCo's `qpos`
  array via the `obj2_joint` freejoint address.
- **`compute_reward`** — evaluates the full shaped reward (grasp + lift + place)
  independently for each object and returns whichever is higher, so the agent gets
  full credit for focusing on either object.
- **`make()` classmethod** — convenience constructor that handles the `set_task`
  boilerplate (required by Meta-World v3 before `step()` is allowed), so the env
  can be instantiated with a single call without going through a benchmark.

### `metaworld/envs/assets_v3/sawyer_xyz/sawyer_bin_picking_two_obj.xml`
Copied from `sawyer_bin_picking.xml` with one addition: a second free body `obj2`
inlined directly (rather than via `<include>`) to avoid duplicate name collisions
with `objA.xml`. The second object uses the same geometry and physics parameters
as obj1 (`bin_base` childclass, `obj_col` geom class) but is colored blue via a
locally defined `obj_blue` material.

```xml
<asset>
    <material name="obj_blue" rgba="0 0.4 1 1" shininess=".2" reflectance="0" specular=".5"/>
</asset>

<body name="obj2" pos="-0.05 0.7 0.04">
    <freejoint name="obj2_joint"/>
    <body childclass="bin_base" name="objB">
        <geom material="obj_blue" size="0.02 0.02 0.02" type="box"/>
        <geom class="obj_col"     size="0.02 0.02 0.02" type="box" mass=".1"/>
    </body>
</body>
```

---

## Modified Files

### `metaworld/envs/__init__.py`
Added one import:
```python
from metaworld.envs.sawyer_bin_picking_v3_two_objects import SawyerBinPickingTwoObjEnvV3
```

### `metaworld/env_dict.py`
Added the new env to the flat environment dict and to the `ALL_V3_ENVIRONMENTS`
list so that all downstream registry checks (`MT1`, `gym.make`, etc.) recognise it:
```python
"bin-picking-two-objects-v3": envs.SawyerBinPickingTwoObjEnvV3,
```

---

## Usage

### Via `gym.make` (standard Meta-World path)
```python
import gymnasium as gym
import metaworld

env = gym.make('Meta-World/MT1', env_name='bin-picking-two-objects-v3', seed=42)
obs, info = env.reset()
```

### Direct instantiation
```python
from metaworld.envs.sawyer_bin_picking_v3_two_objects import SawyerBinPickingTwoObjEnvV3

env = SawyerBinPickingTwoObjEnvV3.make()
obs, info = env.reset()  # obs.shape == (39,)
```

---

## Observation Layout (39 dims)
| Slice     | Content                     |
|-----------|-----------------------------|
| `[0:3]`   | End-effector xyz            |
| `[3]`     | Gripper openness            |
| `[4:7]`   | obj1 xyz                    |
| `[7:11]`  | obj1 quaternion             |
| `[11:14]` | obj2 xyz            ← NEW   |
| `[14:18]` | obj2 quaternion     ← NEW   |
| `[18:]`   | Goal xyz (+ padding)        |