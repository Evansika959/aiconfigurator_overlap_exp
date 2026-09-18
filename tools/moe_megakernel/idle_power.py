#!/usr/bin/env python3
"""P_static(f) on THIS GPU, so the dynamic energy of a kernel can be separated out.

Needed because a slack partition under a deadline does not save static energy by finishing
early -- the chip stays on until the deadline regardless. The quantity that decides
whether slowing it down helps is therefore kappa(f) = P_dyn/f, not total energy.

Measured here rather than imported from overlap_exp so this folder stands alone, and
because that project found two P_static campaigns disagreeing by 3-4 W.
"""
import csv, os, subprocess, time
import torch
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))
CLOCKS = [300, 510, 705, 900, 1050, 1200, 1305, 1410]

subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
torch.cuda.init()
torch.zeros(1, device="cuda")            # context up, so this is idle-with-context
rows = []
try:
    for f in CLOCKS:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{f},{f}"],
                       capture_output=True)
        time.sleep(1.5)                  # let the rail settle before sampling
        with Power(discard=0.8) as p:
            time.sleep(3.0)
        s = p.summary()
        rows.append(dict(clock=f, clock_med=s["clock_med"], idle_w=s["power_w"],
                         n=s["n_samples"]))
        print(f"  {f:4d} MHz  idle {s['power_w']:6.2f} W  (clk read {s['clock_med']})")
finally:
    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
with open(os.path.join(HERE, "data", "idle_power.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("\nwrote data/idle_power.csv")
