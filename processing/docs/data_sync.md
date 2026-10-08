# Pulling new data: Box + Drive -> scored, labelled dataset

**Rule: the lab server's `raw/` tree is a pulled copy and goes stale. When
asked about new or missing data, check upstream — never assume `raw/` is
current.**

Two upstream sources, both reachable through already-configured rclone remotes:

| Source | Remote | Holds |
|---|---|---|
| Google Drive | `gdrive:` | the recordings (XDF + AVI + responses) |
| Box | `box:` | the clinical workbook (`02_PC Data.xlsx`) |

Everything below is automated by **`scripts/sync_study_data.py`**. Read this doc
to understand what it does and to do any step by hand.

```bash
cd processing
export SYNAPSE_DATA_BASE=/data1/anarghya/synapse-data

python -m scripts.sync_study_data --dry-run    # always start here
python -m scripts.sync_study_data              # full run
```

### On a workstation with Google Drive for Desktop: Drive `PC/` IS the data base

Since 2026-10-08 the data base is the Drive `PC/` folder itself. The
coordinator's `Control/` + `Experimental/` are the raw tree, and `processed/`
sits beside them (uploaded from `ub-polar:/data1/anarghya/synapse-data/processed`).
Everything the pipelines write lands in Drive and syncs. Point the tooling at the
mount:

```bash
export SYNAPSE_DRIVE_MOUNT=~/"Google Drive/My Drive/PC-Synapse AudioSight/Participant Data"
export SYNAPSE_DATA_BASE="$SYNAPSE_DRIVE_MOUNT/PC"     # raw = PC/ itself, outputs = PC/processed/
python -m scripts.sync_study_data --dry-run    # prints "drive: mount ..." when active
```

`$SYNAPSE_DRIVE_MOUNT` is the **`Participant Data`** folder (the upload root),
not `PC/`. Box still needs the `box:` rclone remote (it has no mount).
`inventory.raw_root` recognizes the layout: a base with no `raw/` but with
cohort folders is its own raw tree, and cohort folders match case-insensitively
(`Control` here, `control` on the server). With raw on the mount, stage 3 has
nothing to download, so "new" means **not yet in `processed/qc/quality_results.xlsx`**.

- **The server copy is now a snapshot.** Nothing syncs `/data1/anarghya/synapse-data`
  with Drive. To work on the server, pull first (`rclone copy gdrive:PC/processed
  /data1/anarghya/synapse-data/processed --drive-root-folder-id $DRIVE_ROOT`) and
  push back after. Never let both machines write `processed/` at the same time.
- **Make `PC/` available offline** (Finder: right-click > Make available offline).
  Otherwise every pipeline read streams from the network, and the client can
  evict files it has cached.
- **Drive cannot hold symlinks.** `dataset/eeg_only/cohort_20260828_43` was 43
  absolute symlinks into `zero_mask_20260904` plus a pre-migration
  `clinical.csv`; on Drive it holds only that csv. Use `zero_mask_20260904`.
- **Old manifests record `/data1/...` paths** (`root`, `data_root`). They are
  provenance only; no code reads them back.

- **The path is checked by id.** The script reads the `com.google.drivefs.item-id#S`
  xattr Drive for Desktop puts on the folder and refuses anything that isn't
  `DRIVE_ROOT_ID`, so a moved or renamed shortcut can't point it at the wrong tree.
- **Downloads are network-bound.** In streaming mode, reading a file is what
  fetches it, so pulling a subject (~300-500 MB) runs at network speed. An
  interrupted copy deletes the half-copied subject folder, so the next run
  fetches it again rather than treating it as present.
- **Uploads are queued, not done.** Stages 2 and 6 write into the mount, and the
  client syncs them in the background. Check that Drive for Desktop shows no sync
  errors, especially if the mounted account isn't an editor on `PC/`.
  **Each upload replaces the Drive file**: the client creates a new file (new
  id) and trashes the old one, so the previous version lives only in the trash
  (~30 days) and links by Drive id break. Nothing links to them today: the
  `PC_excel Link` doc points at the Box share URL. `rclone copyto` via `gdrive:`
  updates in place and keeps revisions, if that ever matters.

## The steps

| # | Step | Script / command | Notes |
|---|---|---|---|
| 1 | Pull the workbook from Box | `sync_study_data.py` stage 1 | writes `processing/02_PCData.xlsx`; `--commit-workbook` also git-commits that one file |
| 2 | Mirror the workbook to Drive `PC/` | stage 2 | so the coordinator sees labels next to recordings |
| 3 | Download new recordings | stage 3 | diffs `PC/{Control,Experimental}` against `raw/`; skips folders with no `.xdf` |
| 4 | Demographics gate | stage 4 | a subject with no clinical row can't enter a dataset; `--skip-demographics-check` overrides |
| 5 | Score them | `run_quality.py` (stage 5) | writes `processed/qc/quality_results.xlsx` |
| 6 | Publish the QC workbook | stage 6 | uploads to Drive `PC/quality_results.xlsx` |
| 7 | Refresh dataset labels | stage 7 | rewrites each built variant's `labels.csv` |
| 8 | Build the dataset | `pipelines/pair_video.py` then `pipelines/finalize_dataset.py` | **manual** — needs a new cohort config first |

Steps 1–2 always run. `--clinical-only` runs 1, 2 and 7 — the fast path for "the
coordinator entered new demographics, update the datasets", with no download and
no QC. Other flags: `--no-qc`, `--force-qc`, `--no-clinical-refresh`,
`--variant`, `--data-root`, `--date`.

### Why step 7 is not "re-run finalize"

A workbook pull changes the **labels** of an already-built dataset and nothing
else — the epochs and video are untouched. Rewriting `labels.csv` per variant
takes a second; re-running `finalize_dataset` would rebuild ~360 MB of identical
`.fif` per variant. Step 7 mirrors `finalize_dataset._write_clinical` exactly
(same column order, same manifest bookkeeping) so the two cannot drift.

### Step 8 is deliberately manual

New recordings are scored but do **not** enter a dataset automatically: they need
a cohort decision first (which subjects clear the inclusion bar — see
`conf/cohort/` and `docs/variants.md`), and `pair_video` is slow and irreversible.
Make the cohort call, then:

```bash
# 1. add the subjects to the shared paired tree (slow; NB it rewrites build_log.csv
#    for the whole tree, so only the subjects you pass survive in it -- see below)
python -m pipelines.pair_video "cohort.exp=[EXP50,EXP51]" "cohort.ctrl=[]"

# 2. new cohort config listing every subject, then finalize (fast)
python -m pipelines.finalize_dataset preprocessing=zero_mask
```

## Doing it by hand

Both source folders are owned by someone else and are not in this account's
root, so **every path must be rooted by folder id**. Plain `rclone lsd gdrive:PC`
will not find anything.

```bash
DRIVE_ROOT=1Eh9SlATsEUzrGEDgNVfWPLeOWE2Cy90l    # coordinator's upload root
BOX_DIR=324186906035                            # AudioSight Study/01_Participant Data

# what's upstream?
rclone lsd "gdrive:PC/Control"      --drive-root-folder-id $DRIVE_ROOT
rclone lsd "gdrive:PC/Experimental" --drive-root-folder-id $DRIVE_ROOT

# pull one subject (ALWAYS exclude the macOS junk)
rclone copy "gdrive:PC/Experimental/EXP50" \
    /data1/anarghya/synapse-data/raw/experimental/EXP50 \
    --drive-root-folder-id $DRIVE_ROOT --exclude "._*" --exclude ".DS_Store" -P

# pull the workbook
rclone copyto --box-root-folder-id $BOX_DIR "box:02_PC Data.xlsx" 02_PCData.xlsx
```

Drive root: <https://drive.google.com/drive/u/3/folders/1Eh9SlATsEUzrGEDgNVfWPLeOWE2Cy90l>

## Gotchas

**`PC/` is the EEG tree; its siblings are not.** `PC/Control` + `PC/Experimental`
hold one `<PID>/sub-<num>/` per subject with the `.xdf`, `.avi` and
`_responses.csv`. They map onto the local tree with a **name change**:
`PC/Control` -> `raw/control`, `PC/Experimental` -> `raw/experimental`.
`Phone/` (phone pupillometry, `.mp4` + `.json`, no XDF), `DNQ/` (screen
failures), `Practice Data/` and `old_backup/` are **not** mirrored locally.

**The roster differs by modality — never infer enrollment from `PC/` alone.**
`Phone/Control` holds CTRL19, CTRL22–25, CTRL29 and CTRL30: real enrolled
participants with audiometry + phone pupillometry but **no EEG session**, which
is why they appear in neither `PC/` nor the workbook's EEG-oriented sheets. "ID
absent from `PC/`" means "no EEG", **not** "never enrolled".

**Ignore the older top-level `01_Control` / `01_Experimental` Drive folders**
(ids `1p3GulCfl4TJOAnLBkY1nN1D3k1g-Yiu_` / `1fuedMYQX93jnIVU9OM09gJuyULW44i4G`,
hardcoded in `scripts/copy_subjects_to_pc_phone.py`). They are a phone-era
staging area: `01_Control` covers only CTRL10–CTRL25 and its "new" subjects
contain no XDF at all. Diffing against those instead of `PC/` manufactures five
nonexistent controls. `PC/` is authoritative.

**Everything is littered with AppleDouble files.** Uploads come from macOS, so
every folder carries `._*` and `.DS_Store`. Always exclude them — otherwise they
land in the dataset and break folder discovery.

**Address files by id, not name.** The Box workbook has already been renamed once
(`02_PC Data .xlsx` -> `02_PC Data.xlsx`, losing a space). Ids survive renames.

**Compare the workbook by content hash, not mtime** — rclone rewrites mtime on
every copy, so mtime always looks new.

**The Drive-connector MCP tools cannot do this.** `get_file_metadata` on a folder
id works, but `search_files` with `parentId = '<id>'` returns empty for folders
owned by someone else. Use rclone or the Drive for Desktop mount. (`gws`, which
`scripts/download_drive_folder.py` shells out to, is not installed.)

**`quality_results.xlsx` is a single always-current file**, on Drive and locally
— deliberately not dated. `run_quality.py` overwrites the local one, so copy it
aside first if you need to keep a baseline for comparison.

## Workbook layout drift

The coordinator periodically inserts a new condition-flag column, which slides
every measure to the right. This has happened twice: an `ESL` column, then `TBI`
on 2026-09-04 (shifting `Questionnaires (OpenBCi)` +1 from col 9 and
`Audio Data` +1 from col 21).

`synapse_qc/clinical.py` now derives the offset by locating a header label
(`_shift` / `_find_label`) instead of hardcoding positions, so a third insertion
should just work. The `_expect` layout guards still fire on anything else — they
are what caught both insertions instead of silently misreading. **If a guard
trips, find the inserted column and check whether `_shift`'s anchor needs
widening; do not bump the constants by hand.**

Two more workbook quirks the loader handles:

- **Two questionnaire sheets.** The later EXP batch (EXP20, EXP43–46, EXP50–54)
  is entered **only** in `Questionnaires (OpenBCi)`, not the plain
  `Questionnaires` sheet. `load_clinical_rows` reads the primary sheet first and
  falls back to the OpenBCi sheet for subjects it omits, never overriding a
  primary value. Before this, those six cohort subjects had silently blank
  questionnaire scores.
- **`"No Tinnitus"` as a score.** The OpenBCi sheet writes that text in
  `THI_Total` for every control, where the primary sheet leaves the cell blank.
  `_quest_cell` normalizes it to blank, else the column becomes an object dtype
  downstream. A blank `THI_Total` is correct for non-tinnitus subjects.

## Pull history

| Date | Change |
|---|---|
| 2026-08-28 | EXP20's demographics + audiometry (0 -> 75 fields) |
| 2026-09-04 | full demographics + audiometry for EXP50/51/54 (~80 fields each); EXP47 filled in (its bogus `age` of 126.564 became 19). Purely additive — verified no existing value changed across all 43 cohort subjects |

## Status as of 2026-09-04

Fully in sync: 21 CTRL + 36 EXP. EXP50/51/54 were pulled and scored
(Bad/46.7, Good/93.3, Good/86.7) and are **not yet in any built dataset** —
that needs step 8.

EXP50 reproduces EXP52's right-grid fault exactly (all 8 right channels at
SD 0.0000, `corr_bad_frac` 1.000 = Daisy not streaming), so it is left-ear-only
data. The fault is live and intermittent — EXP54 (08-26) and EXP51 (09-01)
recorded a healthy right grid. See `docs/recording_rig_faults.md`.

## Layout note (2026-09-04)

The tree was reorganised by pipeline STEP: `data/` -> `raw/{control,experimental}`
and `outputs/` -> `processed/{qc,eeg,video,paired,dataset,logs}`. Per-
variant files were renamed `clinical.csv` -> `labels.csv` and
`finalize_status.csv` / `pairing_status.csv` -> `build_log.csv`. See
`processed/README.md` in the data tree. `scripts/migrate_layout.py` performed it
and can be read for the exact mapping — note it also rewrote the path columns
inside every `*_alignment.csv`, which a plain `mv` would have silently broken.

**`pair_video` rewrites the whole tree's `build_log.csv` on every run**, scoped
to the cohort you pass it. Adding subjects incrementally therefore truncates that
file to just the new ones; rebuild it from the per-subject `alignment.csv` /
`qc.json` sidecars, or re-run over the full cohort.
