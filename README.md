# Hearing Study Data Collection

PsychoPy experiment and configuration for collecting synchronized EEG, audio, and behavioral data during four auditory tasks. Uses Lab Streaming Layer (LSL) for time-aligned multi-modal recording.

The offline QC and dataset-building pipelines for the recorded data live in [`processing/`](processing/README.md) (formerly the standalone `synapse-data` repo, merged in with history preserved). The raw recordings themselves live on the lab server — see that README's "Data location" section.

## Known hardware issues

- **Channel 13 (R7) is an open circuit** in most recordings — a rig fault,
  not a montage or analysis problem, and the rest of the right grid is fine.
  Diagnosis and repair steps: [`docs/recording_rig_faults.md`](docs/recording_rig_faults.md).
- **Channel 1 (L1) is dead in every session since the 2026-09-17 rewiring**
  (intermittent before that). To check whether the Cyton and Daisy boards
  themselves are working, follow
  [`docs/openbci_board_testing.md`](docs/openbci_board_testing.md).

## Pre-Requisites

- Set up the recording environment by following [this guide](https://docs.google.com/document/d/1NA2v7Z6gLFAqDksrsyBf3V2RNZ6RxAdAVVEvcNDk-yA/edit?usp=sharing)
- Read through the [data collection protocol](https://docs.google.com/document/d/1ouoUjMdvXaoEwp-7u0hbgPy1gG8JQRklimvL0BNJeHc/edit?usp=sharing)
- [Updated setup notes](https://docs.google.com/document/d/1rhVqhTrDCe6JzxTdRydFx3ycqmDQtdOv-aFe-VhdEWo/edit?usp=sharing)

## Tasks

The session runs four tasks in this order: **PMT → HLT → LET → AST**. Each task has its own welcome screen, pre-stimulus fixation, stimulus, post-stimulus fixation, and (for HLT/LET/AST) a response screen.

### PMT — Pupil Muscular Test
Pupil response baseline. Screen color changes; no participant response required.

### HLT — Hearing Loudness Test
Pure-tone sounds at varying frequencies (500 Hz, 4000 Hz) and intensities (20–100 dB). After each tone, the participant rates loudness on a 0–10 scale.

| Tick | Label |
|------|-------|
| 0 | Cannot hear |
| 5 | Clearly audible |
| 10 | Extremely uncomfortable |

Stimuli live in `stimuli/hlt/` (e.g., `tone_60dB_4000Hz.wav`).

### LET — Listening Effort Test
Spoken words played at five SNR levels. The participant types or selects the word they heard.

| Tick | Label |
|------|-------|
| -1 | Unclear (could not understand) |
| 1–20 | The numbered word in the list (three sub-sliders cover ranges 1–6, 7–13, 14–20) |

Stimuli are grouped by SNR level in `stimuli/let/{SNR0,SNR5,SNR10,SNR15,SNR20}/`. Each subfolder contains numbered `.wav` files matching the response options.

### AST — Aversive Sound Test
Naturalistic sounds rated for pleasantness on a 0–10 scale. Each block of 5 trials contains a fixed **3 aversive + 2 neutral** ratio, shuffled within the block. Neutral sounds are filenames prefixed with `control-`; everything else is aversive.

| Tick | Label |
|------|-------|
| 0 | Pleasant |
| 5 | Neutral |
| 10 | Very unpleasant |

Stimuli live in `stimuli/ast/`. The pool must contain at least 3 aversive and 2 neutral files for a full block (otherwise blocks shrink to whatever's available).

## Output

The experiment writes two files per participant:

- `data/<participant>/<participant>_hearing_<timestamp>.csv` — raw PsychoPy trial log (all variables, all routines)
- `exp_data/sub-<participant>/sub-<participant>_responses.csv` — processed responses, one row per stimulus

The processed CSV has the following columns:

| Column | Description |
|--------|-------------|
| `Stim Type` | `HLT`, `LET`, or `AST` |
| `Stim Name` | Filename without extension (e.g., `tone_60dB_500Hz`, `babycry`) |
| `Stim Variant` | HLT → empty; LET → SNR level (`SNR0`–`SNR20`); AST → `aversive` or `neutral` |
| `Repeat Number` | 1-indexed trial number within the task |
| `User Value` | Integer rating. HLT/AST: 0–10. LET: -1 for "Unclear" else the word index. Empty if no response. |
| `User Response Time (s)` | Time from response prompt to click, 3 decimals |

In addition, LSL streams an `xdf` recording per task at `exp_data/sub-<participant>/sub-<participant>_task-<task>_run-<run>.xdf`, with the marker stream `HearingMarkerStream` carrying event labels of the form `<task>_<phase>-<stim_name>` (e.g., `hlt_prestim-tone_60dB_500Hz`, `ast_response-babycry`).

## Configuration

`config.yaml` defines named experimental profiles. Two are bundled:

- `hearing` — production timings (5 s pre/post, 2 s stim, 5 trials/blocks per task)
- `test` — quick smoke test with 1 s timings and 1 block per task

Each profile defines per-task durations and trial counts:

```yaml
hearing:
  prestim: &prestim_value 5         # seconds
  stim: &stim_value 2
  poststim: &poststim_value 5
  pmt_prestim: *prestim_value
  pmt_stim: *stim_value
  pmt_poststim: *poststim_value
  pmt_trials: 5                     # PMT/HLT/LET: total trials; AST: number of 5-stim blocks
  hlt_prestim: 7
  hlt_stim: *stim_value
  hlt_poststim: *poststim_value
  hlt_trials: 5
  let_prestim: *prestim_value
  let_stim: *stim_value
  let_poststim: *poststim_value
  let_trials: 5
  ast_prestim: *prestim_value
  ast_stim: 5
  ast_poststim: *poststim_value
  ast_trials: 5                     # 5 blocks × 5 stim = 25 AST trials (15 aversive, 10 neutral)
```

## Running the Experiment

1. Start LabRecorder and ensure it's listening on `localhost:22345` (required — the experiment opens a TCP socket to LabRecorder at start).
2. Open `hearing.psyexp` in PsychoPy Builder (or run `python hearing.py`).
3. In the startup dialog, set:
   - `participant` — unique numeric ID (auto-generated by default)
   - `run` — run number (usually `1`)
   - `config` — `hearing` or `test`
   - `enable_video` — `true` to record webcam alongside (writes to `exp_data/videos/`)
   - `enable_ppg` — `true` to read PPG over serial (default `false`)

## Repository Layout

```
hearing.psyexp / hearing.py   PsychoPy experiment (Builder source + compiled output)
config.yaml                   Experiment timings and trial counts
eeg_quality/                  Recording-time EEG safety net: pre-flight gate (imported
                              by the experiment) + live watchdog + mock streamer
stimuli/{hlt,let,ast}/        Stimulus audio files actually read at runtime
sounds/                       Raw source library (not read by the experiment directly)
exp_data/                     XDF recordings and processed response CSVs
processing/                   Offline QC + dataset pipelines (former synapse-data repo;
                              own README, docs/, requirements, Hydra conf).
                              Also holds the montage builder (processing/montage/)
                              and exploratory notebooks (processing/notebooks/)
data/                         Raw PsychoPy trial logs (gitignored)
bids_dataset/                 BIDS-formatted EEG exports
```

See `eeg_quality/README.md` for how the pre-flight gate and live watchdog work
and how to run them.
