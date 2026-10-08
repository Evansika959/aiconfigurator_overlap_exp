"""Event-driven power and performance simulator for MoE inference on a GPU whose SMs can
be split into V/f domains.

The model, bottom up:

  demand.py     what a set of experts asks of the hardware, from the routing histogram
  rate.py       how fast it runs: per-SM service time + shared DRAM and L2
  power.py      board power from activity, not from configuration
  hardware.py   the machine: SMs, V/f domains, TDP, limiter, actuation delay
  calib.py      the fitted numbers, one JSON file per GPU
  workload.py   batches of layers with traced routing
  engine.py     the event loop, and the one-layer predictor controllers call
  control.py    who sets the clocks: governor, fixed, tdp_bisection, closed loop, ...

The tools around it:

  run.py              command line: a YAML/JSON scenario in, four tables out
  meas.py             the measured CSVs in ../data, as records
  calibrate.py        fits rate.py and power.py to them
  validate.py         replays every measurement through the simulator
  validate_trace.py   ... and scripted loads that move, on one time axis with the A100
  study.py            the control groups B0-B4: what each ingredient is worth
  sensitivity.py      ... under every assumption a one-domain chip cannot settle
  policies.py         one measured load under many policies
  report.py           the tables of ../RESULTS_SIM.md
  test_sim.py         invariants

Nothing here touches a GPU. See README.md in this directory.
"""
