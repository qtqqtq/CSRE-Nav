# SAC test and analysis scripts

Run these from `SAC/` so relative model and result paths keep their usual
meaning:

```bash
python3 script/test_cspace.py --model interval_cspace_1 --episodes 100
python3 script/profile_sparrow_step.py --steps 300
python3 -u script/run_interval_sweep.py
python3 script/compare_interval_sweep.py
```

`test_cspace.py` evaluates one checkpoint. `profile_sparrow_step.py` measures
simulator step time. `run_interval_sweep.py` trains and tests each interval
setting, and `compare_interval_sweep.py` formats the completed results.
