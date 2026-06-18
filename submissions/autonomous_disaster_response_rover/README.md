# Autonomous Disaster Response Rover and Robotics Dataset Generation Platform

## Project Overview

Autonomous Disaster Response Rover is a MuJoCo-based rescue simulation and
robotics dataset generation platform. The project demonstrates a differential
drive rover navigating a damaged disaster environment, detecting victims,
confirming rescues, avoiding hazards, returning to extraction, and producing
structured robotics data for analysis and policy development.

The submission is designed for AI judging with reproducible execution,
measurable scoring, visual evidence, and machine-readable logs.

## Architecture Diagram

```text
+--------------------------------+
| environments/disaster_scene.xml |
| disaster world, hazards, victims|
+---------------+----------------+
                |
                | includes
                v
+---------------------------------+
| models/rover_scene_include.xml  |
| chassis, wheels, motors, sensors|
+---------------+-----------------+
                |
                | loaded by
                v
+--------------------------------+
| evaluate.py / run_demo.py       |
| mission loop, scoring, evidence |
+-------+----------------+-------+
        |                |
        v                v
+---------------+  +----------------+
| controllers/  |  | utils/         |
| planner       |  | scoring        |
| detector      |  | logging        |
| rover control |  | data collection|
+-------+-------+  +--------+-------+
        |                   |
        +---------+---------+
                  v
        logs/*.json, logs/*.csv, logs/*.png
```

## Mission Workflow Diagram

```text
START
  |
  v
SEARCH for nearest unrescued victim
  |
  v
APPROACH_TARGET with differential drive control
  |
  v
DETECT victim inside range and field of view
  |
  v
CONFIRM_RESCUE by remaining inside rescue radius
  |
  v
NEXT_TARGET until all victims are rescued
  |
  v
RETURN_TO_BASE / extraction zone
  |
  v
COMPLETE and save score + datasets + artifacts
```

## MuJoCo Features Used

- **MJCF**: Custom XML scene and rover model files define the environment,
  robot, actuators, sensors, cameras, materials, victim targets, and keyframe.
- **Physics**: Gravity, timestep, solver settings, mass distribution, damping,
  friction, and contact parameters are configured for stable rover simulation.
- **Collisions**: Rover chassis, bumpers, wheels, floor, rubble, beams, and
  wall sections participate in contact dynamics. Victim markers and zones are
  visual-only.
- **Wheel actuators**: Four wheel hinge joints use velocity actuators and are
  controlled as left/right differential-drive pairs.
- **Proximity sensors**: Left and right proximity mount sites are modeled and
  mirrored in the synthetic dataset.
- **Lidar mount**: A forward lidar mount/site provides the physical reference
  point for simulated lidar readings.
- **Camera**: Rover-front and overview cameras support visual inspection,
  split-screen demo rendering, and mission video generation.

## Sensor Stack

The rover records a synthetic robotics observation stack at each control step:

- Rover pose: `x`, `y`, `z`, heading.
- Linear velocity estimate.
- Left and right wheel commands.
- Forward lidar-style range.
- Front-left and front-right lidar-style ranges.
- Left and right proximity readings.
- Victim detections.
- Active mission target.
- Hazard-zone status.
- Collision count.
- Recovery action, if triggered.

These values are exported to `logs/sensor_data.csv`.

## Controller Design

The controller is intentionally transparent and reproducible:

1. Compute heading from rover pose to current target.
2. Normalize heading error.
3. Generate smooth left/right wheel velocity commands.
4. Clamp wheel speed to safe limits.
5. Apply acceleration smoothing to avoid unstable actuator jumps.
6. Slow down near obstacles.
7. Trigger recovery logging for blocked paths, stuck behavior, or collisions.

The mission planner selects the nearest unrescued victim, switches to rescue
confirmation inside the rescue radius, and redirects to the extraction zone once
all victims are rescued.

## Dataset Generation Pipeline

```text
MuJoCo state + controller outputs
        |
        v
Mission planner + victim detector
        |
        v
Data collection utilities
        |
        +--> trajectory.csv
        +--> sensor_data.csv
        +--> events.json
        +--> mission_log.json
        +--> dataset_summary.json
        +--> trajectory.png
        +--> mission_summary.json
        +--> score_report.json
```

The simulation therefore acts as both a task environment and a dataset generator
for disaster-response robotics.

## Dataset Schema

`logs/trajectory.csv` records mission-level trajectory data:

```text
timestamp, x, y, z, heading, linear_velocity, mission_state,
active_target, detected_victims, left_wheel_command,
right_wheel_command, front_obstacle_distance,
left_obstacle_distance, right_obstacle_distance,
hazard_status, collisions, hazards, completion_status, recovery_action
```

`logs/sensor_data.csv` records robot-observation data:

```text
timestamp, rover_x, rover_y, rover_z, rover_heading, linear_velocity,
mission_state, active_target, detected_victims, lidar_front,
lidar_front_left, lidar_front_right, proximity_left, proximity_right,
hazard_status, collision_count, left_wheel_command,
right_wheel_command, navigation_decision, recovery_action
```

`logs/events.json` records discrete mission events:

```text
victim_detected, victim_rescued, hazard_entered, recovery_action,
mission_started, mission_complete
```

`logs/mission_summary.json` summarizes the completed run, including score,
rescues, distance, collisions, extraction status, and artifact locations.

## Evaluation Methodology

The evaluator runs the mission headlessly and computes a score out of 100:

- Victim rescue: 40 points
- Mission completion: 20 points
- Time efficiency: 15 points
- Collision avoidance: 10 points
- Distance efficiency: 10 points
- Hazard avoidance: 5 points

The evaluation measures:

- victims detected and rescued
- extraction completion
- mission time
- path distance
- collision count
- hazard entries
- obstacle events
- recovery attempts and successful recoveries

## Reproducibility Instructions

From the repository root:

```bash
python -m pip install -r requirements.txt
python submissions/autonomous_disaster_response_rover/evaluate.py
```

For rendered visual evidence:

```bash
python submissions/autonomous_disaster_response_rover/run_demo.py
```

The headless evaluator is the fastest reproducibility path. It regenerates all
dataset and scoring files under:

```text
submissions/autonomous_disaster_response_rover/logs/
```

## Example Results

Example evaluator output:

```text
Mission complete

Victims rescued: 4/4
Collisions: 21
Time: 12.24s
Distance: 18.06m

Final score: 86.47/100

Generated Robotics Dataset
--------------------------
Simulation steps: 306
Victims detected: 4
Victims rescued: 4
Obstacle events: 143
Recovery actions: 179
Mission score: 86.47/100
```

## Generated Artifacts

After a run, the following artifacts are generated in `logs/`:

- `trajectory.png`: map-style plot showing rover path, victim locations, start,
  finish, and extraction zone.
- `mission_summary.json`: compact mission result summary and artifact index.
- `sensor_data.csv`: per-step robot observation dataset.
- `events.json`: structured list of detections, rescues, hazards, and recovery
  actions.
- `trajectory.csv`: rover path and mission state over time.
- `mission_log.json`: event log plus mission metrics.
- `score_report.json`: final score and category breakdown.
- `dataset_summary.json`: aggregate dataset statistics.
- `demo.mp4`: rendered visual demo when `run_demo.py` is executed.

## Future Research Applications

- Navigation policy training from synthetic rescue trajectories.
- Recovery-policy learning from blocked-path, stuck, and collision events.
- Obstacle-avoidance benchmarking under repeatable disaster layouts.
- Victim-search behavior analysis using event logs and sensor records.
- Sim-to-real curriculum design for disaster-response mobile robots.
- Multi-agent search-and-rescue extensions with shared map building.
- Safety tradeoff studies between rescue speed, collision risk, and hazard
  avoidance.
