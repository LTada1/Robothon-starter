ROVER_START_POSITION = [-5.25, -3.75, 0.24]
ROVER_START_YAW = 0.52

EXTRACTION_ZONE = {
    "position": [5.25, 3.65, 0.0],
    "radius": 2.0,
}

SEARCH_ZONES = {
    "search_zone_alpha": {
        "position": [-2.50, -1.20, 0.0],
        "size": [1.10, 0.85],
    },
    "search_zone_bravo": {
        "position": [1.25, 1.05, 0.0],
        "size": [1.20, 0.95],
    },
    "search_zone_charlie": {
        "position": [4.30, 2.25, 0.0],
        "size": [1.25, 1.00],
    },
}

VICTIMS = {
    "victim_1": {
        "position": [-4.05, -2.72, 0.08],
        "detection_radius": 3.0,
        "rescue_radius": 0.8,
        "description": "Easy target near the start zone.",
    },
    "victim_2": {
        "position": [-0.85, 1.28, 0.08],
        "detection_radius": 3.0,
        "rescue_radius": 0.8,
        "description": "Target placed behind debris near the corridor exit.",
    },
    "victim_3": {
        "position": [1.85, -2.52, 0.08],
        "detection_radius": 3.0,
        "rescue_radius": 0.8,
        "description": "Target near the fire hazard area.",
    },
    "victim_4": {
        "position": [4.70, 2.78, 0.08],
        "detection_radius": 3.0,
        "rescue_radius": 0.8,
        "description": "Far target in the final search zone.",
    },
}

HAZARD_ZONES = {
    "hazard_fire_patch": {
        "position": [1.20, -2.25, 0.0],
        "size": [1.05, 0.58],
    },
    "hazard_chemical_spill": {
        "position": [2.95, 0.25, 0.0],
        "size": [0.82, 0.48],
    },
}

OBSTACLES = {
    "corridor_walls": [
        {"name": "left_corridor_wall", "position": [-1.20, -0.18, 0.33], "size": [0.16, 2.15, 0.33]},
        {"name": "right_corridor_wall", "position": [0.15, -0.62, 0.30], "size": [0.16, 1.60, 0.30]},
    ],
    "collapsed_walls": [
        {"name": "collapsed_wall_north", "position": [1.75, 2.15, 0.38], "size": [1.15, 0.16, 0.38]},
        {"name": "collapsed_wall_south", "position": [-3.00, -2.05, 0.34], "size": [1.00, 0.18, 0.34]},
    ],
    "beams": [
        {"name": "steel_beam_1", "position": [-0.45, -2.58, 0.13], "size": [1.05, 0.07, 0.07]},
        {"name": "steel_beam_2", "position": [2.35, -0.65, 0.12], "size": [0.95, 0.06, 0.06]},
        {"name": "steel_beam_3", "position": [3.82, 1.38, 0.11], "size": [0.82, 0.06, 0.06]},
    ],
    "rubble_blocks": [
        {"name": "rubble_block_1", "position": [-3.55, -0.88, 0.16], "size": [0.32, 0.28, 0.16]},
        {"name": "rubble_block_2", "position": [-2.75, 0.25, 0.22], "size": [0.45, 0.24, 0.22]},
        {"name": "rubble_block_3", "position": [-1.92, -1.75, 0.14], "size": [0.28, 0.34, 0.14]},
        {"name": "rubble_block_4", "position": [0.62, 0.78, 0.18], "size": [0.38, 0.30, 0.18]},
        {"name": "rubble_block_5", "position": [1.85, -1.42, 0.20], "size": [0.36, 0.42, 0.20]},
        {"name": "rubble_block_6", "position": [3.05, 2.55, 0.18], "size": [0.40, 0.36, 0.18]},
        {"name": "rubble_block_7", "position": [4.62, 0.68, 0.15], "size": [0.30, 0.44, 0.15]},
        {"name": "rubble_block_8", "position": [3.70, 3.12, 0.20], "size": [0.50, 0.22, 0.20]},
    ],
}
