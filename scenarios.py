import numpy as np


# stores fixed yaw, pitch + insertion configurations in radians + mm
SCENARIOS = {
    "3D detour": {
        "description": "Direct rotation is blocked; use the 3D workspace.",
        "start": np.array([np.deg2rad(20), 0.0, 90.0]),
        "target_configuration": np.array([0.0, 0.0, 90.0]),
        "structure_centre": np.array([65.0, 11.0, 0.0]),
        "structure_radius": 6.0,
    },
    "Direct insertion": {
        "description": "Reach the target with a clear insertion route.",
        "start": np.array([0.0, 0.0, 35.0]),
        "target_configuration": np.array([0.0, 0.0, 90.0]),
        "structure_centre": np.array([65.0, 11.0, 0.0]),
        "structure_radius": 6.0,
    },
    "Retraction required": {
        "description": "Retract before rotating towards the target.",
        "start": np.array([np.deg2rad(-45), 0.0, 90.0]),
        "target_configuration": np.array([np.deg2rad(45), 0.0, 90.0]),
        "structure_centre": np.array([50.0, 0.0, 0.0]),
        "structure_radius": 30.0,
    },
    "Inaccessible target": {
        "description": "The final shaft placement is blocked.",
        "start": np.array([0.0, 0.0, 35.0]),
        "target_configuration": np.array([0.0, 0.0, 90.0]),
        "structure_centre": np.array([65.0, 0.0, 0.0]),
        "structure_radius": 6.0,
    },
}
