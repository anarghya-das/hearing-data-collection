"""Finalize the detect-and-defer paired dataset into a training-ready dataset.

``pipelines/pair_video.py`` does the slow, one-time work: temporal EEG<->video
alignment, filtering, epoching, and video encoding, saving EVERY boundary-valid
trial with all 16 channels intact and bad channels *detected but untouched*
(recorded in each ``_epo.fif``'s ``info['bads']`` + a ``*_qc.json`` sidecar).

This step is the fast, swappable half. For each subject it:
  * applies the channel strategy (``interpolate`` | ``zero_mask`` | ``drop`` |
    ``keep_all``) from the ``preprocessing`` group;
  * emits a per-channel validity mask (1 = real signal, 0 = interpolated/masked/
    dropped) so a model can down-weight non-measured channels;
  * applies PTP z-score epoch rejection over the *good* channels only (so a noisy
    bad channel can't corrupt the rejection), excluding a task when too few epochs
    survive or too many are rejected;
  * filters the paired ``*_alignment.csv`` to the surviving epochs, re-indexing
    ``eeg_epoch_index`` into the finalized ``_epo.fif`` and pointing ``video_clip``
    at the existing clips under ``processed/video`` (video is never
    re-encoded);
  * joins the clinical workbook into ``labels.csv`` (one row per finalized
    subject: questionnaire scores + demographics) so the dataset ships with its
    labels.

Because channel handling is decoupled from the expensive pairing, you can build
``interpolate`` / ``zero_mask`` / ``drop`` variants side by side from one pair run:

    python -m pipelines.finalize_dataset                          # published: interpolate + z=3
    python -m pipelines.finalize_dataset preprocessing=zero_mask
    python -m pipelines.finalize_dataset preprocessing=drop
    python -m pipelines.finalize_dataset preprocessing.epoch_rejection.enabled=false
"""
import os
import sys
import csv
import json
import warnings
from datetime import datetime

import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from synapse_qc import clinical as qclinical, dataset as qdataset, \
    paths as qpaths  # noqa: E402

warnings.filterwarnings("ignore")

_ALIGN_COLS_IN = [
    "task", "trial", "eeg_epoch_index", "condition", "label", "t_stim_lsl",
    "tmin", "tmax", "video_clip", "video_frames_csv", "n_frames_written",
    "n_frames_expected", "partial_window",
]


def _load_qc(eeg_subj, subject_id):
    """Read the per-subject QC sidecar written by pair_video. Returns
    ``(bads, quality_score)`` (bads = [] if no sidecar)."""
    path = os.path.join(eeg_subj, f"sub-{subject_id}_qc.json")
    if not os.path.exists(path):
        return [], None
    with open(path) as fh:
        qc = json.load(fh)
    return list(qc.get("bads_combined", [])), qc.get("quality_score")


def _load_alignment(paired_subj, subject_id):
    """Read the paired alignment rows (empty list if the subject had no video)."""
    path = os.path.join(paired_subj, f"sub-{subject_id}_alignment.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _finalize_subject(pid, eeg_subj, video_subj, paired_subj, out_subj,
                      strategy, rej, montage_file, base=REPO,
                      layout=qdataset.LAYOUT_MATERIALIZED):
    """Finalize one subject; returns a summary row (or None if it has no epochs)."""
    import mne

    eeg_in = eeg_subj
    if not os.path.isdir(eeg_in):
        return None

    # QC sidecars live with the EEG they describe; the trial index lives in the
    # paired tree (and is simply absent for a subject with no video).
    bads, quality_score = _load_qc(eeg_in, pid)
    align_rows = _load_alignment(paired_subj, pid)
    align_by_task = {}
    for r in align_rows:
        align_by_task.setdefault(r["task"], []).append(r)

    os.makedirs(out_subj, exist_ok=True)

    task_counts, excluded_tasks, final_align = {}, [], []
    final_ch_names, final_mask = None, None
    # What the loader replays instead of us writing a second copy of the epochs.
    task_spec = {}

    for fname in sorted(os.listdir(eeg_in)):
        if not fname.endswith("_epo.fif"):
            continue
        task = fname.split(f"sub-{pid}_")[-1].replace("_epo.fif", "")
        epochs = mne.read_epochs(os.path.join(eeg_in, fname), preload=True, verbose=False)

        epochs, mask, ch_names = qdataset.apply_channel_strategy(
            epochs, bads, strategy, montage_file)
        final_ch_names, final_mask = ch_names, mask   # same for every task

        kept, n_before, n_after, excluded, reason = qdataset.reject_epochs(epochs, mask, rej)

        source_rel = os.path.relpath(os.path.join(eeg_in, fname), base)
        if excluded:
            print(f"  [{pid}] {task}: {n_before}->{n_after} — EXCLUDED ({reason})")
            excluded_tasks.append(task)
            task_counts[task] = 0
            task_spec[task] = {"source_file": source_rel, "excluded": True,
                               "reason": reason, "n_source": n_before}
            continue

        task_spec[task] = {"source_file": source_rel, "excluded": False,
                           "kept": [int(k) for k in kept],
                           "n_source": n_before, "n_kept": n_after}
        if layout == qdataset.LAYOUT_MATERIALIZED:
            # Write the transformed epochs so the dataset is directly loadable
            # with mne.read_epochs() and stands on its own.
            eeg_out = os.path.join(out_subj, "eeg")
            os.makedirs(eeg_out, exist_ok=True)
            epochs.save(os.path.join(eeg_out, fname), overwrite=True, verbose=False)
        task_counts[task] = n_after
        print(f"  [{pid}] {task}: {n_before}->{n_after} epochs "
              f"(strategy={strategy}, {int(mask.sum())}/{len(mask)} real ch)")

        # Re-index the paired alignment rows for this task onto the finalized
        # epochs: keep rows whose old epoch index survived, renumber to new index.
        new_index = {old: k for k, old in enumerate(kept)}
        for r in align_by_task.get(task, []):
            old = int(r["eeg_epoch_index"])
            if old not in new_index:
                continue
            video_abs = os.path.join(video_subj, r["video_clip"])
            frames_abs = os.path.join(video_subj, r["video_frames_csv"])
            row = dict(r)
            # eeg_epoch_index indexes the LOADED epochs (row k is row k), exactly
            # as when they were materialised; source_epoch_index indexes the
            # untouched file under processed/eeg/.
            row["eeg_epoch_index"] = new_index[old]
            row["source_epoch_index"] = old
            row["eeg_file"] = (
                os.path.relpath(os.path.join(out_subj, "eeg", fname), base)
                if layout == qdataset.LAYOUT_MATERIALIZED else source_rel)
            row["video_clip"] = os.path.relpath(video_abs, base)
            row["video_frames_csv"] = os.path.relpath(frames_abs, base)
            final_align.append(row)

    if final_ch_names is None:
        return None   # no epoch files at all

    # The variant IS these three files plus the alignment: mask, channel meta,
    # and which source epochs survived.
    with open(os.path.join(out_subj, f"sub-{pid}_epochs.json"), "w") as fh:
        json.dump({"subject_id": pid, "base": base, "strategy": strategy,
                   "montage_file": montage_file, "tasks": task_spec}, fh, indent=2)
    np.save(os.path.join(out_subj, f"sub-{pid}_channel_mask.npy"), final_mask)
    with open(os.path.join(out_subj, f"sub-{pid}_channels.json"), "w") as fh:
        json.dump({"ch_names": final_ch_names,
                   "mask": [int(m) for m in final_mask],
                   "bads_detected": sorted(bads),
                   "strategy": strategy,
                   "quality_score": quality_score}, fh, indent=2)

    # Filtered alignment: one row per surviving paired (EEG epoch, video clip).
    cols = ["task", "trial", "eeg_epoch_index", "source_epoch_index", "eeg_file",
            "condition", "label", "t_stim_lsl", "tmin", "tmax", "video_clip",
            "video_frames_csv", "n_frames_written", "partial_window"]
    if final_align:
        with open(os.path.join(out_subj, f"sub-{pid}_alignment.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(final_align)

    return {
        "subject_id": pid,
        "quality_score": quality_score,
        "n_bad_detected": len(bads),
        "n_real_channels": int(final_mask.sum()),
        "n_channels": len(final_mask),
        "task_counts": task_counts,
        "excluded_tasks": excluded_tasks,
        "paired_trials": len(final_align),
    }


def _load_paired_manifest(paired_root):
    """Provenance of the paired input (cohort, build date); {} if absent."""
    path = os.path.join(paired_root, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _write_quality(out_root, eeg_root, subjects):
    """Ship the QC verdict with the dataset as ``quality.csv``.

    ``build_log.csv`` records *that* channels were masked; this records **why** —
    which criterion flagged each one. The source is each subject's
    ``sub-<PID>_qc.json`` in the EEG tree, i.e. the exact QC run whose
    ``bads_combined`` this variant masked, so the two cannot disagree. (The
    workbook at ``processed/qc/quality_results.xlsx`` carries more per-channel
    detail — SNR, correlation fractions, duplicates — but it is a separate run
    and is not guaranteed to match a given build.)
    """
    from synapse_qc import excel as qexcel   # one definition of the bands

    cols = ["subject_id", "auto_grade", "quality_score", "method", "preset",
            "n_channels", "n_bad", "bads_combined", "bads_flat", "bads_dead",
            "bads_noisy", "bads_lowcorr", "bads_highcorr_reported"]
    rows = []
    for pid in subjects:
        path = os.path.join(eeg_root, pid, f"sub-{pid}_qc.json")
        if not os.path.exists(path):
            rows.append({"subject_id": pid})
            continue
        with open(path) as fh:
            qc = json.load(fh)
        row = {"subject_id": pid,
               # Same Excellent/Good/Average/Bad bands the QC workbook uses --
               # taken from synapse_qc.excel so the two can never disagree.
               "auto_grade": qexcel.auto_grade(qc.get("quality_score"), "ok"),
               "quality_score": qc.get("quality_score"),
               "method": qc.get("method"), "preset": qc.get("preset"),
               "n_channels": qc.get("n_channels"),
               "n_bad": len(qc.get("bads_combined", []))}
        for k in ("bads_combined", "bads_flat", "bads_dead", "bads_noisy",
                  "bads_lowcorr", "bads_highcorr_reported"):
            row[k] = ",".join(qc.get(k, []))
        rows.append(row)
    path = os.path.join(out_root, "quality.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n = sum(1 for r in rows if r.get("quality_score") is not None)
    print(f"[quality] wrote quality.csv ({n}/{len(rows)} subjects with a QC sidecar)")
    return {"csv": "quality.csv", "subjects_with_qc": n}


def _write_clinical(out_root, subjects, cfg):
    """Join per-subject clinical scores + demographics + audiometry from the PC
    workbook into ``labels.csv`` (one row per finalized subject, keyed by
    subject_id) so the multimodal dataset ships with its labels. Parsed directly
    from the workbook by ``synapse_qc.clinical`` (no analysis-repo dependency);
    skipped with a warning if the workbook is unreadable (finalize itself still
    succeeds)."""
    clinical_path = cfg.paths.clinical_data
    if not os.path.isabs(clinical_path):
        clinical_path = os.path.join(REPO, clinical_path)
    measures = list(cfg.clinical.measures)
    try:
        out_rows = qclinical.load_clinical_rows(clinical_path, subjects, measures)
    except Exception as e:  # noqa: BLE001
        print(f"[clinical] SKIPPED — cannot read workbook {clinical_path}: {e}")
        return None

    cols = []
    for r in out_rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    csv_path = os.path.join(out_root, "labels.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    n_scored = sum(1 for r in out_rows
                   if any(str(v).strip() for k, v in r.items()
                          if k not in ("subject_id", "group", "devices_present",
                                       "questionnaire_device")))
    print(f"[clinical] wrote labels.csv ({n_scored}/{len(out_rows)} subjects "
          f"with workbook entries, from {os.path.basename(clinical_path)})")
    return {"csv": "labels.csv", "workbook": clinical_path,
            "measures": measures, "subjects_with_entries": n_scored}


def _write_manifest(rows, out_root, cfg, strategy, rej, variant, paired_manifest,
                    clinical, quality=None, cohort_name="all",
                    layout=qdataset.LAYOUT_MATERIALIZED):
    cols = ["subject_id", "status", "quality_score", "n_bad_detected",
            "n_real_channels", "n_channels", "paired_trials",
            "pmt", "hlt", "let", "ast", "excluded_tasks", "error"]
    with open(os.path.join(out_root, "build_log.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            tc = r.get("task_counts", {}) or {}
            w.writerow({
                "subject_id": r.get("subject_id", ""),
                "status": r.get("status", ""),
                "quality_score": r.get("quality_score"),
                "n_bad_detected": r.get("n_bad_detected"),
                "n_real_channels": r.get("n_real_channels"),
                "n_channels": r.get("n_channels"),
                "paired_trials": r.get("paired_trials", 0),
                "pmt": tc.get("pmt", 0), "hlt": tc.get("hlt", 0),
                "let": tc.get("let", 0), "ast": tc.get("ast", 0),
                "excluded_tasks": ",".join(r.get("excluded_tasks", []) or []),
                "error": r.get("error", ""),
            })
    manifest = {
        "product": "multimodal-final",
        "variant": variant,
        "name": cfg.name,
        "layout": layout,
        "quality": quality,
        "cohort": cohort_name,
        "built": datetime.now().isoformat(),
        "git_sha": qpaths.git_sha(),
        "channel_strategy": strategy,
        "epoch_rejection": rej,
        "inputs": {"paired_dir": cfg.paths.paired_dir,
                   "paired_built": paired_manifest.get("built"),
                   "paired_git_sha": paired_manifest.get("git_sha")},
        "clinical": clinical,
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_failed": sum(1 for r in rows if r.get("status") == "FAILED"),
        "paired_trials_total": sum(r.get("paired_trials", 0) for r in rows),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


@hydra.main(version_base=None, config_path="../conf", config_name="finalize")
def main(cfg: DictConfig) -> None:
    pre = cfg.preprocessing
    strategy = pre.channel_strategy
    rej = OmegaConf.to_container(pre.epoch_rejection, resolve=True)
    # Base for the relocatable outputs trees (montage asset stays repo-relative).
    base = qpaths.resolve_base(cfg.paths.get("root"))
    montage_file = os.path.join(REPO, cfg.paths.montage)
    paired_root = os.path.join(base, cfg.paths.paired_dir)
    eeg_root = os.path.join(base, cfg.paths.get("eeg_dir") or
                            os.path.join(cfg.paths.paired_dir, "_eeg"))
    video_root = os.path.join(base, cfg.paths.get("video_dir") or
                              os.path.join(cfg.paths.paired_dir, "_video"))
    # Variant dir is <cohort>__<preprocessing> (cohort from the paired manifest)
    # so both dimensions of the variant are visible in the path.
    paired_manifest = _load_paired_manifest(paired_root)

    # The eeg/ and video/ trees hold EVERY subject we have processed; a cohort
    # is a DATASET-level choice, so selection happens here. `+cohort=<name>`
    # restricts to that config's ids; with no cohort you get everything.
    cohort = cfg.get("cohort")
    if cohort:
        wanted = set(cohort.get("exp") or []) | set(cohort.get("ctrl") or [])
        cohort_name = cohort.get("name") or "cohort"
    else:
        wanted, cohort_name = None, "all"
    variant = cfg.get("variant") or f"{cohort_name}__{cfg.name}"
    layout = cfg.get("dataset_layout") or qdataset.LAYOUT_MATERIALIZED
    if layout not in (qdataset.LAYOUT_MATERIALIZED, qdataset.LAYOUT_VIEW):
        raise SystemExit(f"dataset_layout must be 'materialized' or 'view', "
                         f"got {layout!r}")

    # Refuse to write a VIEW build into a directory holding a pre-view
    # MATERIALIZED one. The two layouts look similar per-subject but mean
    # different things, and a half-overwritten variant silently mixes stale
    # .fif copies with fresh decision files.
    _out = os.path.join(base, cfg.paths.dataset_dir, variant)
    _old = os.path.join(_out, "manifest.json")
    if os.path.exists(_old):
        with open(_old) as fh:
            prev = json.load(fh)
        if prev.get("layout") == "materialized":
            raise SystemExit(
                f"{variant} is a MATERIALIZED (pre-view) build. Writing a view "
                f"into it would mix layouts. Delete it first, or pass "
                f"variant=<new-name>."
            )
    out_root = os.path.join(base, cfg.paths.dataset_dir, variant)
    os.makedirs(out_root, exist_ok=True)

    print("=" * 70)
    print(f"FINALIZE  variant={variant}  strategy={strategy}  "
          f"reject_enabled={rej.get('enabled')}")
    print(f"  {os.path.relpath(paired_root, base)} -> {os.path.relpath(out_root, base)}")
    print("=" * 70)

    # The EEG tree is the anchor: a subject can legitimately have no video
    # (CTRL06, CTRL21) but never no EEG.
    subjects = sorted(
        d for d in os.listdir(eeg_root)
        if os.path.isdir(os.path.join(eeg_root, d))
    )
    if wanted is not None:
        missing = sorted(wanted - set(subjects))
        subjects = [s for s in subjects if s in wanted]
        if missing:
            print(f"  cohort {cohort_name}: {len(missing)} id(s) not in the EEG "
                  f"tree, skipped: {missing}")
    print(f"  {len(subjects)} subject(s) selected"
          + (f" from cohort {cohort_name}" if wanted is not None
             else " (all processed)"))
    rows = []
    for pid in subjects:
        eeg_subj = os.path.join(eeg_root, pid)
        video_subj = os.path.join(video_root, pid)
        paired_subj = os.path.join(paired_root, pid)
        out_subj = os.path.join(out_root, pid)
        try:
            summary = _finalize_subject(
                pid, eeg_subj, video_subj, paired_subj, out_subj,
                strategy, rej, montage_file, base, layout)
            if summary is None:
                continue
            summary["status"] = "ok"
            rows.append(summary)
        except Exception as e:
            print(f"[error] {pid}: {e}")
            rows.append({"subject_id": pid, "status": "FAILED", "error": str(e),
                         "paired_trials": 0})

    finalized = [r["subject_id"] for r in rows if r.get("status") == "ok"]
    clinical = _write_clinical(out_root, finalized, cfg)
    quality = _write_quality(out_root, eeg_root, finalized)
    manifest = _write_manifest(rows, out_root, cfg, strategy, rej, variant,
                               paired_manifest, clinical, quality, cohort_name,
                               layout)
    print("\n" + "=" * 70)
    print(f"Wrote {os.path.join(out_root, 'build_log.csv')}")
    print(f"  ok={manifest['n_ok']}  failed={manifest['n_failed']}  "
          f"paired_trials_total={manifest['paired_trials_total']}")


if __name__ == "__main__":
    main()
