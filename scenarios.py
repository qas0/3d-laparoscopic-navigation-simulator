import numpy as np


# stores fixed configurations + synthetic vessel centrelines in radians + mm
SCENARIOS = {
    "3D detour": {
        "description": "Direct rotation is blocked; use the 3D workspace.",
        "start": np.array([np.deg2rad(20), 0.0, 90.0]),
        "target_configuration": np.array([0.0, 0.0, 90.0]),
        "structure_points": np.array([[55.0, 8.0, -1.0], [65.0, 7.5, 0.0],
                                      [80.0, 8.0, 1.0], [84.0, 8.25, 1.25]]),
        "structure_radius": 2.5,
    },
    "Direct insertion": {
        "description": "Reach the target with a clear insertion route.",
        "start": np.array([0.0, 0.0, 35.0]),
        "target_configuration": np.array([0.0, 0.0, 90.0]),
        "structure_points": np.array([[55.0, 8.0, -1.0], [65.0, 7.5, 0.0],
                                      [80.0, 8.0, 1.0], [84.0, 8.25, 1.25]]),
        "structure_radius": 2.5,
    },
    "Retraction required": {
        "description": "Retract before rotating towards the target.",
        "start": np.array([np.deg2rad(-45), 0.0, 90.0]),
        "target_configuration": np.array([np.deg2rad(45), 0.0, 90.0]),
        "structure_points": np.array([[48.0, 0.0, -40.0], [49.0, 0.0, -20.0],
                                      [50.0, 0.0, 0.0], [49.0, 0.0, 20.0],
                                      [48.0, 0.0, 40.0]]),
        "structure_radius": 2.5,
    },
    "Inaccessible target": {
        "description": "The final shaft placement is blocked.",
        "start": np.array([0.0, 0.0, 35.0]),
        "target_configuration": np.array([0.0, 0.0, 90.0]),
        "structure_points": np.array([[64.0, 1.0, -12.0], [65.0, 0.0, 0.0],
                                      [66.0, 1.0, 12.0]]),
        "structure_radius": 2.5,
    },
}
