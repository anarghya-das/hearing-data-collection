"""Pair per-trial webcam clips with stim-locked EEG epochs (multimodal dataset).

Ports ``../../synapse/split_video.py`` into this repo's conventions:
  * file resolution via ``synapse_qc.inventory`` (the XDF *and* the per-participant
    ``.avi``), not ``build_combined_mapping`` / next-to-XDF guessing;
  * EEG parameters come from the Hydra ``preprocessing`` group -- the SAME configs
    ``build_dataset`` uses -- so stream / bandpass / notch / channel_strategy /
    quality / epoch_rejection are all configurable and stay in lock-step with the
    processed-pkl variants;
  * task windows from ``tasks.timings``; video options from the ``video`` group;
  * outputs under ``processed/{eeg,video,paired}/`` (one ``<PID>/`` tree each
    + a top-level ``build_log.csv`` and ``manifest.json``).

The published EEG alignment helpers (``create_mne`` and the marker/event
machinery) are reused live from ``../../synapse`` exactly like ``build_dataset``,
with the absolute montage path injected into ``create_mne`` so it does not depend
on the current working directory.

    python -m pipelines.pair_video                                   # usable cohort, published EEG params, epoch mode
    python -m pipelines.pair_video preprocessing=drop                # different bad-channel handling
    python -m pipelines.pair_video video.mode=marker                 # legacy marker-to-marker clips
    python -m pipelines.pair_video preprocessing.bandpass.low=2 preprocessing.notch_freq=-1   # tweak filters (notch<0 disables)
    python -m pipelines.pair_video cohort=published
    python -m pipelines.pair_video cohort.exp='[EXP01,EXP13]' cohort.ctrl='[CTRL10]'  # ad-hoc subset
"""
import os
import sys
import csv
import json
import warnings
from datetime import datetime

import hydra
from omegaconf import OmegaConf, DictConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from synapse_qc import inventory, paths as qpaths  # noqa: E402

warnings.filterwarnings("ignore")


def _patch_epoching(montage_abs):
    """Point the vendored epoching code at this repo's assets and QC.

    Two deliberate interventions, both scoped to pairing:

    1. ``create_mne`` gets the absolute montage path injected, so nothing
       depends on the current working directory.
    2. ``quality_check`` is replaced by :func:`synapse_qc.qc_core.quality_check`
       -- the robust, filtering-invariant, windowed metric. The vendored copy
       still carries the LEGACY mean-correlation rule, which flags healthy
       ear-EEG channels sharing common-mode DC drift as bad (EXP08 / EXP10 /
       CTRL09 / CTRL12 score 0 under legacy but 75-100 under robust).
       ``create_mne`` looks ``quality_check`` up as a module global, so
       rebinding it here routes bad-channel detection through the robust
       metric. The return dict is a superset, so ``create_mne`` consumes it
       unchanged. ``build_dataset`` is deliberately left on the legacy metric,
       because its job is a bit-faithful mirror of the published pkl.
    """
    from synapse_qc import epoching, qc_core

    _orig = epoching.create_mne

    def _create_mne_abs(*a, **k):
        k.setdefault("montage_file", montage_abs)
        return _orig(*a, **k)

    epoching.create_mne = _create_mne_abs
    epoching.quality_check = qc_core.quality_check
    return epoching


def _resolve_cohort(cfg, data_root=None):
    """Yield ``(group, pid, Participant|None)`` for the requested cohort.

    Non-empty list => those IDs. Empty list => auto-discover only when
    ``cohort.name`` is ``all`` (see ``conf/cohort/all.yaml``); otherwise
    empty means none for that group (so CLI ``cohort.exp=[] cohort.ctrl=[CTRL10]``
    runs a single control without pulling in every experimental)."""
    parts = {p.pid: p for p in inventory.discover(data_root=data_root)}

    def _ids(configured, prefix):
        items = list(configured)
        if items:
            return items
        if cfg.cohort.name == "all":
            return [p for p in parts if p.startswith(prefix)]
        return []

    want = {
        "EXP": _ids(cfg.cohort.exp, "EXP"),
        "CTRL": _ids(cfg.cohort.ctrl, "CTRL"),
    }
    for grp, ids in want.items():
        for pid in ids:
            yield grp, pid, parts.get(pid)


def _write_dataset_manifest(rows, out_root, cfg, eeg_params):
    """Top-level CSV + JSON capturing per-subject status and the run provenance."""
    csv_path = os.path.join(out_root, "build_log.csv")
    cols = [
        "group", "subject_id", "status", "mode", "n_bad_detected", "quality_score",
        "video_present", "paired_trials", "pmt", "hlt", "let", "ast",
        "excluded_tasks", "error",
    ]
    flat = []
    for r in rows:
        tc = r.get("task_counts", {}) or {}
        flat.append({
            "group": r.get("group", ""),
            "subject_id": r.get("subject_id", ""),
            "status": r.get("status", ""),
            "mode": r.get("mode", cfg.video.mode),
            "n_bad_detected": r.get("n_bad_detected"),
            "quality_score": r.get("quality_score"),
            "video_present": r.get("video_present"),
            "paired_trials": r.get("paired_trials", 0),
            "pmt": tc.get("pmt", 0), "hlt": tc.get("hlt", 0),
            "let": tc.get("let", 0), "ast": tc.get("ast", 0),
            "excluded_tasks": ",".join(r.get("excluded_tasks", []) or []),
            "error": r.get("error", ""),
        })
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)

    manifest = {
        "product": "multimodal-paired",
        "name": cfg.name,
        "cohort": cfg.cohort.name,
        "built": datetime.now().isoformat(),
        "git_sha": qpaths.git_sha(),
        "mode": cfg.video.mode,
        "eeg_params": eeg_params,
        "tasks": OmegaConf.to_container(cfg.tasks.timings, resolve=True),
        "video": OmegaConf.to_container(cfg.video, resolve=True),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_failed": sum(1 for r in rows if r.get("status") == "FAILED"),
        "n_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "paired_trials_total": sum(r.get("paired_trials", 0) for r in rows),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return csv_path, manifest


@hydra.main(version_base=None, config_path="../conf", config_name="pair_video")
def main(cfg: DictConfig) -> None:
    # Base for the relocatable data/outputs trees (assets stay repo-relative).
    base = qpaths.resolve_base(cfg.paths.get("root"))
    data_root = (cfg.paths.get("data_root") or os.environ.get("SYNAPSE_DATA_ROOT")
                 or inventory.raw_root(base))

    montage_abs = os.path.join(REPO, cfg.paths.montage)
    _patch_epoching(montage_abs)
    from synapse_qc import av_align

    pre = cfg.preprocessing
    vid = cfg.video
    out_root = os.path.join(base, cfg.paths.paired_dir)
    eeg_root = os.path.join(base, cfg.paths.get("eeg_dir")
                            or os.path.join(cfg.paths.paired_dir, "_eeg"))
    video_root = os.path.join(base, cfg.paths.get("video_dir")
                              or os.path.join(cfg.paths.paired_dir, "_video"))
    for d in (out_root, eeg_root, video_root):
        os.makedirs(d, exist_ok=True)

    # Pairing is detect-and-defer: it applies stream/bandpass/notch/quality
    # (detection) but NOT channel_strategy or epoch_rejection -- those are consumed
    # downstream by finalize_dataset, over this full paired set. channel_strategy is
    # still passed for the legacy marker mode.
    eeg_kw = dict(
        eeg_stream_name=pre.eeg_stream_name,
        channel_strategy=pre.channel_strategy,
        bandpass=OmegaConf.to_container(pre.bandpass, resolve=True),
        flat_voltage=pre.quality.flat_voltage,
        notch_freq=pre.notch_freq,
        quality_preset=pre.quality.preset,
        bindings=list(vid.bindings),
        task_timings=OmegaConf.to_container(cfg.tasks.timings, resolve=True),
        mode=vid.mode,
        exclude_phases=list(vid.exclude_phases),
        fps=vid.fps,
        sfreq=vid.sfreq,
        no_eeg=vid.no_eeg,
        no_video=vid.no_video,
    )

    print("=" * 70)
    print(f"PAIR VIDEO  name={cfg.name}  cohort={cfg.cohort.name}  mode={vid.mode}")
    print(f"  EEG: stream={pre.eeg_stream_name}  bp={eeg_kw['bandpass']}  "
          f"notch={pre.notch_freq}  quality={pre.quality.preset}  "
          f"(detect-only; channel_strategy + rejection deferred to finalize)")
    print(f"  -> {os.path.relpath(out_root, base)}")
    print("=" * 70)

    rows = []
    for grp, pid, p in _resolve_cohort(cfg, data_root):
        row = {"group": grp, "subject_id": pid, "mode": vid.mode}
        if p is None or not p.xdf_path:
            print(f"[skip] {pid}: no resolvable XDF")
            row.update({"status": "skipped", "error": "no resolvable XDF",
                        "paired_trials": 0})
            rows.append(row)
            continue
        try:
            summary = av_align.pair_recording(
                p.xdf_path, p.video or None, pid,
                eeg_dir=os.path.join(eeg_root, pid),
                video_dir=os.path.join(video_root, pid),
                align_dir=os.path.join(out_root, pid),
                **eeg_kw
            )
            row.update({"status": "ok", **summary})
        except Exception as e:  # isolate one subject's failure (e.g. no EEG stream)
            print(f"[error] {grp}/{pid}: {e}")
            row.update({"status": "FAILED", "error": str(e), "paired_trials": 0})
        rows.append(row)

    # Record what pairing actually applied vs. what it deferred, so the manifest
    # documents that channel_strategy + epoch_rejection are finalize's job.
    eeg_params = {
        "applied": {k: eeg_kw[k] for k in (
            "eeg_stream_name", "bandpass", "flat_voltage",
            "notch_freq", "quality_preset")},
        "deferred_to_finalize": {
            "channel_strategy": pre.channel_strategy,
            "epoch_rejection": OmegaConf.to_container(pre.epoch_rejection, resolve=True),
        },
    }
    csv_path, manifest = _write_dataset_manifest(rows, out_root, cfg, eeg_params)

    print("\n" + "=" * 70)
    print(f"Wrote {csv_path}")
    print(f"  ok={manifest['n_ok']}  failed={manifest['n_failed']}  "
          f"skipped={manifest['n_skipped']}  "
          f"paired_trials_total={manifest['paired_trials_total']}")


if __name__ == "__main__":
    main()
