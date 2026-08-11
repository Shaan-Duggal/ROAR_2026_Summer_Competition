# ROAR Simulation Racing — Summer 2026 submission

`competition_runner.py` and `infrastructure.py` in this repository are the **unmodified official files**. 

Solution entry point: [`competition_code/submission.py`](competition_code/submission.py).


## Special instructions

**None.** There is nothing to install beyond the standard competition environment.

`submission.py` only imports `numpy` and `roar_py_interface`, both of which the official `competition_runner.py` and `infrastructure.py` already require. 

Run it exactly as the [documentation](https://roar.gitbook.io/roar-competition-documentation/general-submission-instructions) describes:

```sh
conda activate roar_competition
cd competition_code
python competition_runner.py
```

## Contents

```
README.md                      this file
competition_code/
    submission.py              entry point (RoarCompetitionSolution)
    ThrottleController.py      longitudinal control
    LateralController.py       steering
    WaypointLine.py            racing-line lookup
    SectionStats.py            per-section bookkeeping
    SpeedData.py               speed model container
    waypoints/
        waypointsPrimary.npz   the racing line
        location_with_radius   precomputed corner radii
    competition_runner.py      official, unmodified
    infrastructure.py          official, unmodified
```
