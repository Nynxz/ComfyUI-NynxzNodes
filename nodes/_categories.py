"""The pack's node-menu layout, in one place. Named after core's menus, so a node sits where its
core counterpart does. A category is only where a node is listed: node ids never change with it."""

from ._zenkit import CATEGORY as ROOT

LOADERS = f"{ROOT}/Loaders"
OUTPUT = f"{ROOT}/Output"
CONDITIONING = f"{ROOT}/Conditioning"
FUSION = f"{CONDITIONING}/Fusion"
REGIONS = f"{ROOT}/Regions"
FRESCO = f"{REGIONS}/Fresco"
SAMPLING = f"{ROOT}/Sampling"
MASK = f"{ROOT}/Mask"
IMAGE = f"{ROOT}/Image"
GRID = f"{IMAGE}/Grid"
VIDEO = f"{ROOT}/Video"
UTILS = f"{ROOT}/Utils"
DIAGNOSTICS = f"{ROOT}/Diagnostics"
