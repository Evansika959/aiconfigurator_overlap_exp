# Overnight results

Three GPU campaigns and the analyses that follow from them. Every figure regenerated;
`data/README.md` is the index.

---

## 1. Comm-bound sweep — the gap that mattered, now measured

`data/commbound_{128,256,512}mib.csv`, 1080 rows. 6 shapes x 3 message sizes x 4 CTA
counts x 6 clocks, world=4.

**Why it was needed.** Every two-clock number until tonight rested on 64 MiB, where a
sensibly chosen CTA count leaves 11 of 12 shapes GEMM-bound. Only 23% of configurations
reached `t_comm/t_gemm > 1`, and mostly through 1-2 CTA settings no planner would pick.
This sweep reaches **86%**, median ratio 5.7.

### Two-clock gain, measured

Every input is a measurement. Only the composition of two solo kernels into one
overlapped iteration is inferred, and in this regime it validates at **0.60% median**
against the measured `concurrent` rows — six times tighter than the 3.94% on 64 MiB data,
because the collective dominates and there is less for the re-waving term to get wrong.

| `t_comm/t_gemm` | n | median | best |
|---|---|---|---|
| < 1 | 9 | +2.43% | +0.61% |
| 1 – 3 | 10 | **−7.24%** | −18.57% |
| 3 – 10 | 28 | **−10.48%** | −22.42% |
| > 10 | 19 | **−19.89%** | **−23.17%** |

Overall median −10.07%; 47 of 66 configurations save more than 5%. The chosen settings
are strikingly uniform: one clock is forced to 1200–1410, two clocks put the **GEMM at
300–510 and the collective at 1410**.

### Two by-products

**`B(c,f)` extrapolates cleanly past its calibration ceiling.** Fitted on 8–256 MiB, at
512 MiB it lands at **0.96% median error** — tighter than the 1.11% *within* calibration.
Every earlier claim involving a large message was extrapolating; that is now checked.

**The composition rule is far more accurate here** (0.60% vs 3.94%), so the comm-bound
numbers are the best-supported in the whole study.

---

## 2. Can an overlap throttle when neither kernel alone does?

**No — 0 of 492**, and now with a mechanism rather than an absence.

`data/throttle_gap.csv`, 142 rows. Shapes built specifically to sit near the boundary.

Measured on 30 pairs at 1410 MHz:

| | |
|---|---|
| change in mean power from adding the collective | median **−178 W** |
| smallest reduction | −60 W |
| cases where it **raises** mean power | **0 of 30** |

The collective is far less power-dense than the GEMM, so time-sharing dilutes the
average. If the GEMM alone does not throttle, overlapping it cannot push it over —
**the overlap moves power the wrong way**. The earlier 0/432 was not the coverage
artifact I assumed; it was the answer.

### My shape design failed, and the reason is worth keeping

All three engineered shapes throttled on their own:

| M | S_eff | predicted | measured | held |
|---|---|---|---|---|
| 1024 | 64.0 | 321 W | 333.0 | yes |
| 1792 | 74.7 | 360 W | 394.6 | **no**, 1380 |
| 1280 | 80.0 | 380 W | 398.4 | **no**, 1365 |
| 1536 | 96.0 | 438 W | 400.0 | **no**, 1290 |

I extrapolated from two anchors in the 432 data — M=1024 at 321 W and M=2048 at 399 W —
but **M=2048 was already throttled**. 399 W is the cap, not what that GEMM would have
drawn. A line through one free point and one clamped point has a flattened slope.

Corrected: between S_eff 64 and 75 the GEMM goes from 333 W to the cap. The 340–395 W
window is about **six SMs wide**. The gap in the original data was the hardware's, not
the shape grid's.

---

## 3. Voltage knee resolved — 1020–1080 MHz, not 900

`gemm_squat_knee.csv`, 480 rows, 0 failures. Four clocks added inside the 900–1200 gap on
the squatter rig, so `a` is identifiable rather than merely bounded.

| f | 300 | 510 | 705 | 900 | **960** | **1020** | **1080** | **1140** | 1200 | 1410 |
|---|---|---|---|---|---|---|---|---|---|---|
| `kappa` ×10⁻³ | 1.689 | 1.670 | 1.675 | 1.674 | **1.721** | **1.743** | **1.851** | **2.008** | 2.219 | 3.170 |
| vs floor | 1.007 | 0.996 | 0.999 | 0.998 | 1.026 | 1.039 | **1.104** | 1.197 | 1.323 | 1.890 |

Flat to 0.5% CV through **1020 MHz**, then climbing. The knee is **120 MHz higher** than
the six-clock grid suggested.

### What it did NOT fix

Refitting barely moved the throttled-row rescore: median 5.05% → **4.72%**, and the
clock-correlated residual is **unchanged at +0.93**. So that residual is not an
interpolation artifact.

I then tested a second explanation — that `clock_min` is a biased stand-in for the
time-average clock. The recorded `med − min` gap on throttled rows is only **15 MHz**
median (max 45), and its correlation with throttle depth is −0.28, "no clear trend".
That accounts for 1–2%, not a −5% bias.

**Two explanations tested, neither accounts for it. It stays open.** I am not proposing
a third without testing it.

### A discrepancy I am not papering over

Two campaigns disagree about `P_static`:

| f | p0_static campaign | voltage_knee idle | diff |
|---|---|---|---|
| 900 | 61.36 W | 64.34 W | +2.98 |
| 1200 | 69.66 W | 74.13 W | +4.47 |

One forces the P0 state with zeus's method; the other locks the clock and reads idle.
3–4 W is ~5% of the static floor, which is itself ~47% of board power.

The **knee location is safe** — `kappa` takes both terms from the same campaign, so a
constant offset cancels. **Absolute power is not safe to mix**: splicing the new idle
values into a model validated against p0_static would shift every number and silently
invalidate the 3.3% figure. So the new `a(f)` is adopted and `P_static` keeps its
original campaign. Reconciling the two is open.

---

## 4. A bug that silently dropped 78 measurements

The 128 MiB pass lost its **entire 16-CTA batch** to `EADDRINUSE`. Free-port selection
binds port 0, reads the number, closes, and hands it to the workers — anything can take
it in that window, and the loop printed one line and moved on.

The window cannot be closed (the children must be able to bind), so the fix is to detect
a startup death and **retry with a new port**, up to four times. Backfilled; 128 MiB is
now complete at 384 rows.

---

## New figures

| | |
|---|---|
| `fig6_knee.png` | the voltage curve with the gap filled, knee marked |
| `fig7_commbound.png` | two-clock gain vs `t_comm/t_gemm`, measured, by message size |

---

## What I would do next

**The comm-bound result is the strongest thing here** — measured inputs, 0.60%
composition error, 47/66 configurations saving >5%, a clean monotone trend. It is also
the first number in this study that is not mostly model.

**The open items, in order:** the unexplained clock-correlated residual; the two
disagreeing `P_static` campaigns; and `B(c,f)` above 32 CTAs, which several optima still
sit against as a bound rather than an optimum.
