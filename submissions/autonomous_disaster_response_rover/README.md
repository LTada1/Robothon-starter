# Autonomous Disaster Response Rover

Autonomous Disaster Response Rover is a MuJoCo rescue simulation and robotics data
collection environment. A differential-drive rover searches a disaster scene,
detects victim markers, confirms rescues, returns to extraction, and produces
measurable evidence for AI judging.

## How to Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python submissions/autonomous_disaster_response_rover/evaluate.py
```

For rendered visual evidence:

```bash
python submissions/autonomous_disaster_response_rover/run_demo.py
```

## Mission

The rover starts in a marked start zone, navigates through rubble, collapsed
walls, beams, a narrow corridor, hazard patches, and three search zones. It must
rescue four victims and return to the extraction zone.

## Scoring

The final score is out of 100:

- Victim rescue: 40
- Mission completion: 20
- Time efficiency: 15
- Collision avoidance: 10
- Distance efficiency: 10
- Hazard avoidance: 5

## Robotics Data Collection

This simulation generates synthetic disaster-response robot data for navigation
policy training, analysis, and evaluation. Each evaluation run logs rover state,
wheel commands, simulated lidar/proximity readings, victim detections, obstacle
events, hazard status, collision events, recovery actions, trajectory, and final
score.

Generated dataset files:

```text
logs/mission_log.json
logs/events.json
logs/trajectory.csv
logs/sensor_data.csv
logs/score_report.json
logs/dataset_summary.json
```

The dataset can be used to inspect autonomous rescue behavior, compare
navigation decisions against obstacles, and evaluate recovery behavior in a
repeatable disaster environment.
