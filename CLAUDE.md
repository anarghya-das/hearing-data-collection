# CLAUDE.md

Notes that aren't obvious from reading the code. Record the *why*.

## Repo shape: collection at the root, processing under `processing/`

This repo holds BOTH halves of the SYNAPSE hearing study: the PsychoPy
data-collection experiment (repo root) and the QC/dataset-building pipelines
(`processing/` — the former standalone `synapse-data` repo, merged in via
`git subtree` with its history preserved). `processing/` has its own README,
requirements, Hydra `conf/` and `docs/`; run its pipelines FROM INSIDE
`processing/` (`cd processing && python -m pipelines.build_dataset ...`).
The data lives in the Drive `PC/` folder (since 2026-10-08; the lab server's
`/data1/anarghya/synapse-data` is now an unsynced snapshot), pointed at via
`$SYNAPSE_DATA_BASE` — a base holding the recordings (`Control/` +
`Experimental/` on Drive, `raw/{control,experimental}` on the server) and
`processed/` (everything generated, numbered by pipeline step: `qc` -> `paired`
-> `dataset`). Layout and the reserved
`eeg`/`video` slots for pupillometry: `processing/README.md` and
`processed/README.md` in the data tree. The
analysis/paper repo is still separate: `../synapse` (a sibling of THIS repo's
root; configured as an absolute path in `processing/conf/config.yaml`
`paths.synapse_repo`, so nothing breaks if cwd differs).

## Upstream data is on Drive/Box — `data/` is a stale pulled copy

**When asked about new or missing data, check upstream — never assume the lab
server's `data/` tree is current.** The coordinator uploads recordings to Google
Drive and maintains the clinical workbook on Box; both are reachable via the
configured `rclone` remotes (`gdrive:` / `box:`), and both MUST be addressed by
folder id, not path. On a workstation with Google Drive for Desktop, Drive can
instead go through the mount (`--drive-mount` / `$SYNAPSE_DRIVE_MOUNT`, checked
against the folder id via the client's xattr); Box always needs rclone.

`processing/scripts/sync_study_data.py` does the whole chain (workbook pull ->
mirror to Drive -> download new recordings -> demographics gate -> QC -> publish
-> refresh dataset labels). Run `--dry-run` first.

**Read `processing/docs/data_sync.md` before touching any of this** — it has the
steps, the folder ids, the by-hand commands, and the traps that will otherwise
cost you an hour (the non-EEG sibling folders, the phone-only subject ids that
look like missing controls, the stale partial `01_Control` staging folder, the
macOS `._*` litter, and why the MCP Drive tools can't enumerate these folders).

## Processing (`processing/`) — QC rules

`processing/` was the standalone `synapse-data` repo (subtree-merged, history
preserved). Run its pipelines from inside it. Layout, output paths and env vars
are in `processing/README.md`; do not restate them here.

- **Two EEG streams per recording — pick deliberately.** `obci_eeg1` is RAW,
  `obci_eeg2` is the OpenBCI GUI's filtered stream (~5-50 Hz + 60 Hz notch).
  **QC scores the RAW stream**; the filtered one is a cross-check
  (`score_filtered`) only. Which stream looks "good" flips per participant, so
  never auto-pick best-of-two as the headline.
- **`robust` is the default metric, `legacy` exists only for comparison.**
  Robust band-passes a copy to 1-50 Hz before every criterion, so the score is
  filtering-invariant. Legacy computes on the unfiltered signal and flags
  common-mode DC drift as bad, scoring healthy recordings 0 (CTRL09: legacy 0,
  robust 81). The correlation criterion is windowed (PREP-style) at 10 %, NOT
  PREP's 1 % — cEEGrid sits over jaw muscle, so healthy channels briefly
  decorrelate. Rationale + empirics: `docs/QC_methodology_review.md`,
  `docs/QC_grade_bands_review.md`.
- **Use *max* inter-channel correlation, never mean.** Ear-EEG channels are
  weakly correlated with the contralateral ear, so a healthy channel's mean
  correlation is naturally low.
- **Duplicate detection runs on RAW, never the filtered copy.** Band-passing
  strips the shared DC and collapses the separation: after 1-50 Hz, duplicated
  CTRL27 (0.984) scores BELOW healthy CTRL10 (0.9996).
  See `docs/recording_rig_faults.md`.
- **R07 is excluded from the score denominator, not from `bad_channels`.** It is
  a rig fault (open circuit, 88 % of recordings), so charging every participant
  for it is wrong; it is still detected and still masked downstream. **Remove
  `exclude_from_score` once the hardware is repaired**, else a regression hides.
- **Never re-reference before QC** — and the pipeline never re-references at
  all (`grep set_eeg_reference` returns nothing; keep it that way). A common
  average HIDES bad channels (CTRL12: 66.7 -> a perfect 100 under CAR while 5
  channels rail for ~45 % of the recording); bipolar montages fail the opposite
  way. Re-referencing clean data is fine; re-referencing to DECIDE what is clean
  is not. Evidence: `docs/QC_methodology_review.md`.
- **`quality_score` is "% channels surviving QC", not signal fidelity.**
  `100·(1 − n_bad/16)`. Don't over-read small differences.
- **Detect the device by STREAM name, not folder name.** OpenBCI (16ch) is
  scored; Neurable MW75 (14ch) is not. EXP32 is Neurable-only and its folder
  name does NOT say so — use `qc_core.detect_devices`.
- **Per-participant resolution quirks belong in `synapse_qc/inventory.py`**, in
  one place: `-old` sessions skipped unless they are the only one (CTRL01/02/03),
  EXP44's stream empty, EXP26/CTRL03 have no EEG stream.
- **The clinical loader fails loudly on misaligned questionnaire rows** (GAD vs
  `Severity Data Sum`; it caught CTRL06-09). Fix such rows on Box, not in code;
  see `processing/docs/data_sync.md` "Workbook layout drift".

## Processing — pipelines and datasets

- **TWO `quality_check` implementations, on purpose — do not unify them.**
  `epoching.quality_check` is the vendored LEGACY metric and is the default,
  because `build_dataset cohort=published` must mirror the published pkl.
  `qc_core.quality_check` is the ROBUST one used by every QC entry point;
  `pair_video` rebinds to it at run time, `build_dataset` deliberately does not
  (rebinding changes bad channels on 23/28 published subjects).
- **This repo owns processing**; the analysis repo (`../synapse`) only reads a
  finished pkl. The published preprocessing code is vendored verbatim — full
  rationale and the re-vendoring procedure in `docs/dataset_handoff.md`.
- **Two stages, detect-and-defer.** `pair_video` writes the stage trees
  (`processed/{eeg,video,paired}/`) recording bad channels but NOT
  interpolating/dropping/zeroing or rejecting epochs; `finalize_dataset` applies
  `channel_strategy` + epoch rejection into `processed/dataset/`. This is why
  rejection PTP can be computed over good channels only, and why you can try
  masking vs interpolation without re-running the slow pair step.
- **The stage trees hold EVERY processed subject — no cohort subfolder.**
  Choosing a subset is a dataset decision: `+cohort=<name>` on finalize, omitted
  for all. Layout and the three shipped label files: `processing/README.md` and
  `processed/README.md` in the data tree.
- **`docs/variants.md` is GENERATED from `conf/` — never hand-edit it.**
  `python -m synapse_qc.variants --write` regenerates, `--check` exits 1 on
  drift. Run `--check` after touching `conf/cohort/` or `conf/preprocessing/`.
- **Two clinical workbooks, per-cohort on purpose.** `cohort=published` pins the
  older `../synapse/02_PC Data.xlsx` so the mirror reproduces bit-faithfully;
  every other cohort uses the newer local `02_PCData.xlsx`. Don't "fix" the
  published cohort to use the new workbook.
- **The prior hand ratings are deliberately NOT in this repo** so the first QC
  pass stays independent; `synapse_qc/manual.py` can load them for comparison.
- Use `bun` (not npm/npx) for JS tooling; Context7 MCP for library docs.

## `hearing.psyexp` is the source of truth — never hand-edit the generated files

`hearing.py` and `hearing_lastrun.py` are **generated by PsychoPy Builder** from
`hearing.psyexp`. Builder overwrites them on every compile / Run, so any manual
edit to them is silently lost. **Only edit `hearing.psyexp`.** Do not bother
editing `hearing_lastrun.py` (it is regenerated when the user opens the psyexp
in Builder and presses Run).

The experiment's runtime Python lives **inside Code Components** in the psyexp,
not in a normal `.py`. The recording/LSL/LabRecorder logic is in the `code_init`
component (`Before Experiment` = imports + class defs; `Begin Experiment` = the
LabRecorder `update`/`select all`/`start` sequence).

### Editing the psyexp safely

- It is XML; the Python is stored entity-encoded inside `val` attributes:
  newlines as `&amp;#10;`, `"` as `&quot;`. A literal `\n` in code stays as
  backslash-n.
- PsychoPy is **not installed in this repo's tooling env**, so you can't
  recompile to validate. After any psyexp edit: re-parse with ElementTree,
  decode `&#10;`→newline, and `compile()` the affected Code Component blocks.
- Prefer verified raw-text transforms (read file, `.replace(old, new)` with a
  `count == 1` assertion) over the Edit tool for multi-line encoded blocks.
- **Builder can clobber external edits**: if the user has `hearing.psyexp` open
  in Builder and it re-saves, it overwrites edits made on disk. Make psyexp
  edits with Builder closed, and have the user re-open it fresh afterwards.

## OPEN HARDWARE FAULTS in the recording rig (4 of them)

The right cEEGrid's **R7** pad — OpenBCI **channel 13**, the Daisy's 5th input —
is electrically disconnected in most sessions (bad in 38/43 live recordings).
In the failing sessions it reads a constant **-187500.0156 uV**, exactly the
ADS1299 negative rail at gain 24, which is what a *floating* input does. It is
an OPEN, not a short and not a gel problem. The rig, not the montage: R5 (the
other immediate neighbour of GND at R6) is healthy, and excluding ch13 the
right grid matches the left (22.3% vs 18.6% bad, p=0.86) — so **the right-side
data is usable**. Do not "fix" this in analysis code. Diagnosis + repair
procedure: `docs/recording_rig_faults.md`.

THREE MORE, same doc: (2) the 7 "all 16 channels dead" recordings are ONE
REF/BIAS failure each -- every channel pinned at the negative rail -- not 16
dead electrodes; REF is L6 (LEFT grid) and BIAS is R6 (RIGHT grid) -- OPPOSITE
connectors, so either socket can cause it. NB the reference design contradicts
itself on this: its "Channel Selection" section says L6=REF/R6=GND while
"Connecting the Adapter PCB" says R4a/R4b. L6/R6 is correct -- it is the only
one consistent with the repo's own channel table (3 and 6 skipped on BOTH ears
= 8+8). Do not trust the R4a/R4b line. (3) **EXP47
and CTRL27 contain DUPLICATED L/R data** -- all 8 pairs at r~1.000000, unity
scale, so the right grid carries no independent signal; QC scored them Good/87.5
and Average/62.5 because `bads_highcorr` is reported but NEVER SCORED. Treat
them as 8-channel or exclude. (4) EXP52 AND EXP50 have the whole Daisy block at
literal 0.0 in both streams = Daisy not streaming; suspect the Y-splitter. This
one is LIVE AND INTERMITTENT: EXP50 (tested 2026-08-19) reproduces EXP52's
signature exactly -- all 8 right channels at SD 0.0000, corr_bad_frac 1.000 --
while EXP54 (08-26) and EXP51 (09-01) recorded a healthy right grid. So it is a
flaky connection, not a dead Daisy board, and it is still losing half the
montage in new sessions. Check the splitter before each run.

Gotcha when re-deriving it: the QC workbook shows ch13 SD as `0.0000`, but that
is the POST-BAND-PASS SD (a DC constant filters to zero variance). The raw value
is a rail. Reading it as "zero" flips the diagnosis from open to short.

## EEG recording safety net (`eeg_quality/`)

Two decoupled layers guard against silently losing a session's EEG/video:

- **Pre-flight gate** (`eeg_quality/recording_checks.py`) runs *inside* the
  experiment. It must stay importable in the **PsychoPy app bundle**, which
  ships only numpy + pylsl — so it is deliberately mne-free and does only
  structural/flow checks (stream resolves, samples flowing, not frozen). The
  Builder imports it as `from eeg_quality.recording_checks import ...`;
  `eeg_quality/__init__.py` is intentionally empty so that import never pulls in
  mne. It runs **before** any recorder thread starts and before LabRecorder's
  `start`, so a failed check leaves no XDF (and no video file) behind. Video is
  checked only when `enable_video` is true.
- **Live watchdog** (`eeg_quality/eeg_watchdog.py`) is a *separate* window run in
  an env that has mne (`eeg_quality/requirements.txt`). It uses the project's
  real `qc_core` robust QC to catch what the minimal gate can't: a montage that
  is connected but railed, and a stream that dies mid-session.

`eeg_quality/qc_core.py` is a **thin re-export** of the canonical
`processing/synapse_qc/qc_core.py` (it was a vendored copy back when processing
was the separate `synapse-data` repo — nothing to re-sync anymore). It loads
the file by path instead of importing the `synapse_qc` package because the
package `__init__` imports pandas-needing modules the watchdog env lacks.
Offline/post-hoc QC of recorded files lives in `processing/`, not here.

### Why the gate can't score signal quality in-process

PsychoPy's bundled Python (`/Applications/PsychoPy.app/Contents/Resources/lib/
python3.10/`) is write-protected and ships numpy/scipy/matplotlib without pip
metadata; installing mne there would overwrite them (and pull numpy 2.x),
breaking PsychoPy. Hence the split: mne-based scoring only ever runs outside the
bundle (the watchdog / analysis repo).
