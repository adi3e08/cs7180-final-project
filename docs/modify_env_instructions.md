# Multi Object Bin Picking Environment

## Goal
Extend Meta-World's `bin-picking-v3` environment to include more objects.

## Locate the metaworld folder inside your conda environment
It is probably located at a path like this `~/miniconda3/envs/cs7180/lib/python3.13/site-packages/metaworld`

## New Files

1. Add the files `sawyer_bin_picking_v3_two_objects.py` and `sawyer_bin_picking_v3_two_objects.py` to the folder `metaworld/envs`
2. Add the files `sawyer_bin_picking_two_obj.xml` and `sawyer_bin_picking_three_obj.xml` to the folder `metaworld/assets/sawyer_xyz/`

## Modified Files

### `metaworld/envs/__init__.py`
Add this along with other imports at the top of the file:
```python
from metaworld.envs.sawyer_bin_picking_v3_two_objects import SawyerBinPickingTwoObjEnvV3
from metaworld.envs.sawyer_bin_picking_v3_three_objects import SawyerBinPickingThreeObjEnvV3
```

Add this to the list `__all__`
```python
    "SawyerBinPickingTwoObjEnvV3",
    "SawyerBinPickingThreeObjEnvV3"
```

### `metaworld/env_dict.py`
Add this to the `ENV_CLS_MAP` dict:
```python
    "bin-picking-two-objects-v3": envs.SawyerBinPickingTwoObjEnvV3,
    "bin-picking-three-objects-v3": envs.SawyerBinPickingThreeObjEnvV3,
```

Add this to the `ALL_V3_ENVIRONMENTS` list:
```python
  "bin-picking-two-objects-v3",
  "bin-picking-three-objects-v3",
```