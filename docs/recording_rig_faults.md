# Recording rig faults — diagnosis and repair

Four distinct hardware faults found in the OpenBCI + cEEGrid rig, in order of
severity. Fault 1 (R7) is the most frequent; **fault 3 is the most serious** —
it silently produced duplicated data that the QC scored as good.

**Status:** open, not yet repaired. **Affects:** OpenBCI recordings from
2025-02-27 onward. **Owner:** whoever next has the rig on the bench.

**Update 2026-10-08.** The rig was rewired on 2026-09-17: separate Cyton/Daisy
references (L6 / R6, no Y-splitter), bias moved to L5, channel 4 moved to L3.
R7 is still dead afterwards, and **L1 (channel 1)**, intermittent since mid-2025,
is now dead in every session. To test whether the boards themselves work before
chasing the wiring, use [`openbci_board_testing.md`](openbci_board_testing.md).

## First: the reference design contradicts itself about REF/GND — use L6/R6

`MKnierim/openbci-ceegrids` states the REF/GND assignment **two different ways**:

| Section | Statement |
|---|---|
| *Channel Selection* | "excluding channel L3 & R3, and using channel **L6 as REF and R6 as GND**" |
| *Connecting the Adapter PCB* | "connect the designated electrodes/pins (here **R4a and R4b**) to the bottom SRB (for REF) and bottom BIAS (for GND)" |

**L6/R6 is the correct one for this layout**, because only it is consistent with
the repo's own channel table (Cyton = L1,L2,L4,L5,L7,L8,L9,L10; Daisy =
R1,R2,R4,R5,R7,R8,R9,R10 — i.e. 3 *and* 6 skipped on both ears):

- `L6 = REF, R6 = GND` -> 10 per ear, minus L3/R3 excluded, minus L6/R6 for
  REF/GND = **8 + 8 = 16**. Matches the table exactly.
- `R4a/R4b` -> the right grid would lose R3 + R4a + R4b = 7 recorded and the left
  only L3 = 9 recorded. Asymmetric, and position 6 would not be skipped.
  **Contradicts the table.** That line appears to be carried over from the
  standard Debener cEEGrid convention, where R4a/R4b *are* the usual ref/ground
  pair — it does not describe this adapter's default layout.

**Consequences, which differ from R4a/R4b in ways that matter here:**

1. **REF and GND sit on OPPOSITE grids** — REF on the left, BIAS on the right.
   So the whole-board rail of fault 2 can be caused by **either** connector, not
   just the right one. Two single points of failure on two different sockets.
2. **R5 and R7 are both immediate neighbours of R6 (GND)** — and behave
   completely differently: R5 is 20.9 % bad at a normal 10.87 uV median SD,
   R7 is 88.4 % bad at 0.00 uV. Two electrodes equidistant from GND, opposite
   outcomes. This *strengthens* the conclusion that R7 is a channel-specific
   hardware fault, not any kind of reference/ground proximity effect.
3. **L5 and L7 are the immediate neighbours of L6 (REF)**, and they do show the
   expected physics: median SD 9.00 uV versus 11.70 uV for the other left
   channels. A channel next to the reference has a smaller differential signal.
   That is normal, not a fault — but the QC applies **absolute** thresholds
   (`flat_voltage` 0.5 uV, `sd_floor_uv` 0.3 uV) uniformly, so REF-adjacent
   channels are structurally closer to being flagged. Worth keeping in mind
   before reading anything into L7's elevated 32.6 % failure rate.

---

## Fault 1 — R7 open circuit (most frequent)

The right-ear cEEGrid electrode **R7** — OpenBCI **channel 13**, the Daisy
board's 5th input — is electrically disconnected in most sessions. This is a
mechanical fault in the recording rig, not a data-analysis artefact and not a
problem with the right hemisphere of the montage.

---

## 1. The evidence

Found while auditing per-channel QC failure rates across all 54 participants
(`processed/qc/quality_results.xlsx`).

**It is not random.** Failure rates differ strongly by electrode position
(chi-square = 95.6, p < 0.0001), and one channel dominates:

| Position | Failure rate (43 live recordings) |
|---|---|
| **R7** | **88.4 %** (38/43) |
| R1 | 34.9 % |
| L7 | 32.6 % |
| … | … |
| L9 | 4.7 % |

**It is a hard rail, not a weak signal.** In the failing sessions channel 13
reads a *single constant value for every sample*: **-187500.0156 uV**. That is
-0.1875 V, exactly the ADS1299 negative full-scale rail at gain 24
(+/- Vref/gain = +/- 4.5/24 = +/- 187.5 mV), matching to the LSB. An ADS1299
input pinned at full scale is the signature of **nothing connected to it** —
input bias current charges the floating pin until the converter saturates.

> Note for anyone re-deriving this: the QC workbook reports channel 13's SD as
> `0.0000` in these sessions. That is the *post-band-pass* SD — a constant DC
> value filters to zero variance. Do not read it as "the channel reads zero";
> the raw value is a rail. This distinction is what separates an **open**
> circuit (rail) from a **short** to reference (true zero), and it inverts the
> diagnosis.

**The amplifier channel itself is fine.** In the periods when it does work,
channel 13 records normally — e.g. EXP47 channel 13 SD 8260 uV versus its
neighbour channel 12 at 7165 uV. So the ADS1299 input and the firmware are
healthy; only the connection to it fails.

**It is intermittent in month-long blocks**, which is why a mechanical cause is
likely rather than a per-session gel/prep mistake:

| Period | Channel 13 behaviour |
|---|---|
| 2025-02 .. 2025-06 | railed (open) |
| 2025-06 .. 2025-09 | live, often very noisy |
| 2025-09 .. 2026-02 | railed (open) |
| 2026-06 .. 2026-08 | mixed |



---

# Fault 2 — whole-board rail: a single REF/BIAS failure, not 16 dead electrodes

The 7 recordings previously written off as "all 16 channels dead" are not 16
independent electrode failures. **Six of them have every channel pinned at the
NEGATIVE rail** (-187.5 mV), which is what happens when the reference or bias
connection is lost: every input floats and saturates together.

| Recording | mean samples at rail | channels >90 % railed | sign |
|---|---|---|---|
| CTRL11, CTRL13, CTRL15, CTRL16 | 100 % | 16/16 | all negative |
| EXP12 | 98.8 % | 16/16 | all negative |
| CTRL14 | 93.5 % | 15/16 | all negative |
| **EXP14** | **36.9 %** | **0/16** | **mixed — a different fault** |

REF is **L6 (left grid)** and BIAS is **R6 (right grid)** — they sit on
*opposite* connectors, so a single bad connection at **either** socket takes
down all 16 channels. This reframes six lost sessions as one recurring
single-point failure, but it does **not** localise it to the right grid: check
L6 -> Cyton SRB and R6 -> Cyton BIAS continuity, both.

CTRL01 and CTRL02 are the intermediate case: all 8 right-grid channels railed
while the left stayed clean, i.e. REF/BIAS still contacting but the right grid's
recording electrodes not.

**Diagnostic:** treat any recording where *all* channels rail as a REF/BIAS
connection failure. Check R4a -> Cyton SRB and R4b -> Cyton BIAS continuity
first, before suspecting electrodes.

---

# Fault 3 — DUPLICATED L/R data (most serious; QC did not catch it)

**EXP47 (2026-06-23) and CTRL27 (2026-06-25) contain no independent right-grid
data.** Every left channel and its right counterpart are the same signal:

| | EXP47 | CTRL27 |
|---|---|---|
| L/R pairs with r ~ 1.000000 | 8/8 (L01/R01 bit-identical) | 8/8 |
| amplitude scale R/L | 1.0000 | 1.0000 |
| R-L offset | 0.00-0.49 uV | 0.02-0.55 uV |

Correlations of 1.000000 at unity scale with sub-microvolt offsets are two ADCs
sampling **the same physical inputs** — not a software copy (which would be
bit-identical throughout) but a wiring fault, with the Daisy inputs seeing the
left grid's electrodes. The two sessions are two days apart, so this was a
wiring state during that period; check whether anything was re-plugged around
2026-06-23.

**The QC missed this completely.** EXP47 scored **Good / 87.5** with only 2 bad
channels; CTRL27 scored **Average / 62.5**. Both are in the usable tier. The
`bads_highcorr` ("bridge?") check is the only thing that looks for this and it
is **reported, never scored** — and for CTRL27 it did not fire at all.

**Action:** treat EXP47 and CTRL27 as **8-channel recordings**, or exclude them.
Do not use their right-grid channels. Re-check any session recorded between
2026-06-20 and 2026-06-30.

**Now implemented.** `qc_core.find_duplicate_channels` flags any pair at
|r| >= `duplicate_corr` (0.9999) and scores the later channel of each pair as
bad. **It must run on the UNFILTERED signal** — band-passing removes the shared
DC/drift component and relatively amplifies each ADC's own noise, destroying the
separation. Measured across this dataset:

| | duplicated | healthy |
|---|---|---|
| max \|r\| on raw | **1.000000** | <= 0.997 |
| max \|r\| after 1-50 Hz | 0.984 (CTRL27) | 0.9996 (CTRL10) |

After band-passing, a duplicated recording scores *lower* than a healthy one.
Raw is the only place the two classes separate.

**Seven recordings carry duplicated channels** (2026-08-28 run):

| Recording | dup channels | grade before -> after |
|---|---|---|
| EXP47 | R02,R04,R05,R07,R08,R09,R10 | Good 87.5 -> **Bad 46.7** |
| CTRL27 | R01,R02,R04,R05,R07,R10 | Average 62.5 -> **Bad 46.7** |
| CTRL06 | L05,R04,R05,R08 | -> Bad 40.0 |
| CTRL14 | L10,R10 | Bad 0 (already dead) |
| EXP06 | L10 | Average 60.0 |
| EXP08 | R08 | Average 73.3 |
| CTRL12 | L08 | Average 66.7 |

EXP47, CTRL27 and CTRL06 lose whole blocks and should be excluded or treated as
reduced-channel recordings. The single-channel cases (EXP06, EXP08, CTRL12) are
one duplicated pad each, now correctly scored bad.

---

# Fault 4 — Daisy not streaming (EXP52, the most recent session)

**EXP52 (2026-08-25)** is the newest recording and is a different failure again.
Channels 9-16 read **literal 0.0** on every sample in *both* `obci_eeg1` and
`obci_eeg2` — not the rail. Literal zeros across the whole Daisy block mean the
Daisy was not delivering data at all, rather than its inputs floating.
Separately, L01 is railed at 100 % (a fault-1-type open on the Cyton side).

The two sessions before it (EXP43 2026-08-04, EXP53 2026-08-10) both scored Good
/ 87.5, so this appeared after 2026-08-10 and is a **live problem on the current
rig**.

**Diagnostic:** check the **Y-splitter cable** between Cyton and Daisy (the
reference design requires it and it is the single point of failure for the
entire right block), that the Daisy is seated, and that the GUI/firmware is in
16-channel mode. The wiggle test applies here at the Y-splitter and board-stack
level rather than at an individual pin.

---

# Also: R7 clips at the rail even in the recent "good" sessions

R7 is not merely absent-or-present. In the two most recent healthy recordings it
is intermittently saturating:

| Recording | samples at negative rail, ch13 |
|---|---|
| EXP43 (2026-08-04) | 79.3 % |
| EXP53 (2026-08-10) | 62.6 % |

Both scored Good / 87.5 overall, and R7 was flagged bad in both — but "flagged
bad" understates it: the channel is clipping for most of the recording. This is
consistent with a high-impedance / marginal connection rather than a clean open.

---

## Fault 1 details — what was ruled out, and why

- **Not proximity to the reference or ground.** GND is R6 (see the section
  above on the reference design's contradictory REF/GND text). **R5 and R7 are
  its two immediate neighbours** — R5 is completely healthy (median SD 10.87 uV,
  20.9 % bad) while R7 is 88.4 % bad at 0.00 uV. Equidistant from GND, opposite
  outcomes, so proximity explains nothing. REF is on the *other* grid (L6)
  entirely.
- **Not the reference electrode.** All 16 channels reference to R4a; if REF
  were compromised every channel would degrade together. The other 15 do not.
- **Not a solder bridge / short.** A short to REF or a disabled channel yields
  a value at or near *zero*. This channel rails. (An earlier draft of this
  analysis proposed a bridge at the edge-card socket; the raw-sample evidence
  above supersedes it.)
- **Not gel.** A gel-starved electrode gives high impedance and a noisy trace,
  not full-scale saturation for an entire recording.
- **Not a channel-label mix-up.** The pipeline's `CEEGRID_CH_LABELS` matches the
  reference design's mapping exactly — Cyton ch 1-8 = L1,L2,L4,L5,L7,L8,L9,L10
  and Daisy ch 9-16 = R1,R2,R4,R5,R7,R8,R9,R10 — so channel 13 really is R7.

**Conclusion: an open circuit somewhere between the R7 pad and the Daisy's
channel-5 input.**

## 3. Impact on the collected data — the right side is usable

Excluding channel 13, the right grid is statistically indistinguishable from
the left, and carries slightly *more* signal:

| Measure | Left grid | Right grid | p |
|---|---|---|---|
| Bad-channel rate, **including** ch13 | 18.6 % | 30.5 % | 0.002 |
| Bad-channel rate, **excluding** ch13 | 18.6 % | **22.3 %** | **0.86** |
| Median channel SD | 10.9 uV | **13.0 uV** | 0.06 |

The entire apparent "right ear problem" was this one input. **7 of the 8
right-grid electrodes are good and those recordings are usable.** ADS1299
channels are independent, so a railed channel does not contaminate its
neighbours — which the table above confirms empirically.

Two separate incidents, not caused by this fault:

- **CTRL01 (2025-02-27) and CTRL02 (2025-02-28)** — whole right grid dead, the
  first two sessions of the study, consecutive days. Reads as a setup learning
  curve.
- **EXP52 (2026-08-25)** — whole right grid reads literal `0.0` on every right
  channel (not a rail), i.e. the Daisy side not connected at all. Its own
  incident; investigate separately since it is the most recent recording.

## 4. Diagnostic procedure

**Safety:** battery power only, never mains-tethered. Never put a multimeter on
electrodes attached to a person. Do all continuity testing with the grid off the
participant and the board powered down.

The path being bisected:

```
R7 pad -> cEEGrid flex tail -> mini edge card socket -> adapter PCB trace
       -> jumper wire -> Daisy channel-5 pin (OpenBCI channel 13)
```

Verify pin numbering against your board's silkscreen before heating anything;
channel 13 is the Daisy's 5th input, but confirm physically.

### Step 1 — Rule out software (2 min)
In the OpenBCI GUI confirm channel 13 is enabled and gain is 24 (consistent with
the +/-187.5 mV rail measured). A *disabled* channel reads flat zero, not a
rail, so this is expected to come back clean — eliminate it anyway.

### Step 2 — Wiggle test (highest yield for an intermittent open)
Gel the grid on a test surface (or forearm), stream the GUI time-series view,
and gently flex — one at a time — the flex tail where it enters the socket, the
jumper wire, and the socket body, watching channel 13. If the trace pops between
railed and live at a specific point, that is the fault; skip to Step 5.

### Step 3 — Impedance check
Run the GUI impedance measurement. Channel 13 should read open / off-scale while
the others sit in the 10-50 kOhm range the reference design cites. Record the
numbers so they can be compared after repair.

### Step 4 — Swap test (partitions grid vs. rig, no soldering)
Swap the two cEEGrids — right-ear grid into the left socket and vice versa.

| Result | Conclusion |
|---|---|
| Fault **follows the grid** (appears on a Cyton channel) | The cEEGrid is damaged: bad R7 pad or broken flex trace |
| Fault **stays on channel 13** | The adapter, socket, jumper or Daisy pin is at fault |

Given the fault persists across many participants and presumably several grids,
the second outcome is expected — confirm rather than assume.

### Step 5 — Continuity test to localise the open
Multimeter in continuity mode, board powered off. Probe hop by hop:

| From | To | Expect |
|---|---|---|
| R7 pad on the grid | corresponding socket contact | < a few Ohm |
| socket contact | adapter PCB pad/via | < a few Ohm |
| adapter pad | far end of the jumper wire | < a few Ohm |
| jumper end | Daisy channel-5 header pin | < a few Ohm |

Whichever hop reads open is the fault. Also probe R7's pin **against its
neighbours and against SRB/BIAS** — there should be *no* continuity; if there
is, there is a bridge as well as an open.

## 5. Repair

- **Open at the socket or PCB** — reflow the joint. The reference design's
  advice to remove every second pin of the edge-card socket before soldering
  lowers bridging risk but leaves joints mechanically fragile, which is a
  plausible origin here. Add strain relief (hot glue or epoxy) at the socket
  afterwards; this is the point that flexes in use.
- **Open in the jumper** — replace the wire, routing it *below* the Cyton board
  as the reference design specifies.
- **Damaged grid** — retire that cEEGrid. Review the cleaning routine: the
  reference design warns against rubbing the pads or using chemicals on grids
  intended for reuse.

## Current QC results (2026-08-28 run, all 54 participants)

With the windowed correlation criterion, duplicate detection, and R07 excluded
from the score:

| | count |
|---|---|
| Excellent | 5 |
| Good | 22 |
| Average | 8 |
| Bad | 15 |
| No EEG / empty / other device | 4 |

Median score 80.0, mean 66.7, computed over `n_scored` = 15 channels.
Cohort under the study's own rule (>= 4 good channels): **28 EXP / 15 CTRL = 43**.
At a score >= 60 quality bar: 25 EXP / 10 CTRL = 35.

## 6. Verify the fix

Record 2-3 minutes with the grid on, then from `processing/`:

```bash
python run_quality.py --only <TEST_ID> --data-root /data1/anarghya/synapse-data/data
python spotcheck.py <TEST_ID>
```

Channel 13 should be absent from the bad-channel list with an SD in the same
range as its neighbours (roughly 8-17 uV in the healthy recordings) — neither
near zero nor in the hundreds.

## 7. Follow-up code changes (not yet done)

- **Absolute rail check in QC.** EXP10's channel 13 passed QC at 1197 uV SD,
  which is plainly railed. The robust-z `noisy` criterion missed it because z is
  computed against that recording's own median/MAD, and a heterogeneous
  recording widens the MAD enough to hide an extreme outlier. An absolute upper
  amplitude bound alongside the relative one would catch this class of fault.
- **Pre-flight rail check at collection time.** `eeg_quality/recording_checks.py`
  runs before LabRecorder starts and is deliberately numpy-only; a "no channel
  sitting at +/- full scale" test fits it cleanly and would have caught this on
  day one instead of eighteen months later.
- **Channel 13 is now excluded from the quality-score denominator.**
  `exclude_from_score: ['R07']` in every preset in `qc_core.py`. R07 is still
  detected, still listed in `bad_channels`, and still dropped/masked downstream
  — it is only removed from the SCORE, so a rig defect is not charged to every
  participant. The workbook records `n_scored` (15) and `excluded_from_score`
  for transparency. **Remove this exclusion once the fault is repaired**, and
  re-run `run_quality.py`; leaving it in place after a fix would hide a
  regression.

## References

- Reference design: <https://github.com/MKnierim/openbci-ceegrids>
- Knierim, M. T., Schemmer, M., & Bauer, N. (2022). A simplified design of a
  cEEGrid ear-electrode adapter for the OpenBCI biosensing platform.
  *HardwareX, 12*, e00357. <https://doi.org/10.1016/j.ohx.2022.e00357>
- Per-channel QC data: `processed/qc/quality_results.xlsx` (Per-Channel sheet)
- Related: `processing/docs/QC_grade_bands_review.md`
