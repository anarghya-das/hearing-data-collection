"""Build a processed-data variant: cohort (participant IDs) x preprocessing.

Uses this repo's VENDORED copy of the published preprocessing code
(synapse_qc/process.py + synapse_qc/epoching.py), so a `preprocessing=published`
build still faithfully mirrors how synapse_preprocessed.pkl was made --
`pipelines/compare.py` is the regression test for that. Nothing is imported from
the analysis repo at runtime: this repo owns processing, and the analysis repo
only reads the finished pickle.

The only thing we change from the published flow is FILE RESOLUTION: we map
participant IDs -> XDF via synapse_qc.inventory (which correctly falls back to
the `-old` folder for CTRL01/02/03, whose folders were renamed to `-old` after
the pkl was built; the published `exclude -old` discovery would now drop them).

    python -m pipelines.build_dataset cohort=published preprocessing=published
    python -m pipelines.build_dataset --multirun preprocessing=published,interpolate,zero_mask
    python -m pipelines.build_dataset cohort.exp='[EXP01,EXP13]' cohort.ctrl='[CTRL10]'
"""
import os
import sys
import json
import pickle
import warnings
from datetime import datetime

import hydra
from omegaconf import OmegaConf, DictConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from synapse_qc import inventory, paths as qpaths  # noqa: E402
from synapse_qc import clinical as qclinical, process as pp  # noqa: E402

warnings.filterwarnings("ignore")


def _patch_epoching(montage_abs):
    """Inject the absolute montage path into the vendored ``create_mne`` so it
    does not depend on the current working directory.

    Note ``quality_check`` is deliberately NOT replaced here (``pair_video``
    does replace it): this pipeline's job is a bit-faithful mirror of the
    published pkl, which was built with the legacy metric.
    """
    from synapse_qc import epoching

    _orig = epoching.create_mne

    def _create_mne_abs(*a, **k):
        k.setdefault("montage_file", montage_abs)
        return _orig(*a, **k)

    epoching.create_mne = _create_mne_abs


def _resolve_cohort(cfg, data_root=None):
    """Return ({EXP: {pid: xdf}}, {CTRL: {pid: xdf}}, skipped[]) for the cohort."""
    parts = {p.pid: p for p in inventory.discover(data_root=data_root)}
    want = {
        "EXP": list(cfg.cohort.exp) or [p for p in parts if p.startswith("EXP")],
        "CTRL": list(cfg.cohort.ctrl) or [p for p in parts if p.startswith("CTRL")],
    }
    mapping = {"EXP": {}, "CTRL": {}}
    skipped, resolved = [], {}
    for grp, ids in want.items():
        for pid in ids:
            p = parts.get(pid)
            if p is None or not p.xdf_path:
                skipped.append({"pid": pid, "reason": "no resolvable XDF"})
                continue
            mapping[grp][pid] = p.xdf_path
            resolved[pid] = {"sub_dir": p.sub_dir, "old_only": p.old_only,
                             "neurable": p.is_neurable}
    return mapping, resolved, skipped


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pre = cfg.preprocessing
    variant = cfg.variant
    print("=" * 70)
    print(f"BUILD DATASET  variant={variant}  cohort={cfg.cohort.name}  "
          f"channel_strategy={pre.channel_strategy}")
    print("=" * 70)

    # Base for the relocatable data/outputs trees (assets stay repo-relative).
    base = qpaths.resolve_base(cfg.paths.get("root"))
    data_root = (cfg.paths.get("data_root") or os.environ.get("SYNAPSE_DATA_ROOT")
                 or inventory.raw_root(base))

    montage_abs = os.path.join(REPO, cfg.paths.montage)
    _patch_epoching(montage_abs)

    # Load the (published) global event-ID map directly for consistent event codes.
    with open(os.path.join(REPO, cfg.paths.global_event_map), "rb") as f:
        global_event_map = pickle.load(f)

    task_order = list(cfg.tasks.order)
    task_timings = OmegaConf.to_container(cfg.tasks.timings, resolve=True)
    quality_thresholds = OmegaConf.to_container(pre.quality, resolve=True)
    epoch_rejection_config = OmegaConf.to_container(pre.epoch_rejection, resolve=True)
    channel_strategy = pre.channel_strategy

    mapping, resolved, skipped = _resolve_cohort(cfg, data_root)
    print(f"Resolved EXP={len(mapping['EXP'])} CTRL={len(mapping['CTRL'])} "
          f"| skipped={len(skipped)}")

    exp = pp.process_group(mapping["EXP"], global_event_map, "EXP",
                           task_order, task_timings, quality_thresholds,
                           epoch_rejection_config, channel_strategy=channel_strategy)
    ctrl = pp.process_group(mapping["CTRL"], global_event_map, "CTRL",
                            task_order, task_timings, quality_thresholds,
                            epoch_rejection_config, channel_strategy=channel_strategy)

    # Clinical / behavioural / report keys, built with the published builders so the
    # pkl is a drop-in for the CURRENT synapse analysis scripts (which read
    # channel_masks, clinical_scores, etc.). Schema mirrors run_preprocessing's
    # save_preprocessed dict exactly.
    clinical_measures = list(cfg.clinical.measures)
    # Workbook choice is per-cohort: cohort.clinical_data (published pins the
    # old ../../synapse copy for reproduction fidelity) falls back to
    # paths.clinical_data (the newer local 02_PCData.xlsx). Relative paths
    # resolve against the repo dir, not Hydra's run cwd.
    clinical_path = cfg.cohort.get("clinical_data") or cfg.paths.clinical_data
    if not os.path.isabs(clinical_path):
        clinical_path = os.path.join(REPO, clinical_path)
    clinical_data = pp.load_clinical_data(clinical_path, clinical_measures)
    if not clinical_data:
        print(f"WARNING: clinical_data is EMPTY (could not read {clinical_path}); "
              f"clinical_scores/demographics will be empty.")
    all_subjects = list(exp["subjects"]) + list(ctrl["subjects"])
    clinical_scores = {s: pp.extract_clinical_scores(clinical_data, s, clinical_measures)
                       for s in all_subjects}
    demographics = {}
    for s in all_subjects:
        demo = pp.extract_demographics(clinical_data.get("demographics"),
                                       clinical_data.get("audio"), s)
        if demo:
            demographics[s] = demo
    responses = {}
    responses.update(pp.load_responses(mapping["EXP"], group_name="EXP"))
    responses.update(pp.load_responses(mapping["CTRL"], group_name="CTRL"))
    quality_report = pp.generate_quality_report(exp, ctrl, task_order)

    # Top-level keys mirror the current synapse save_preprocessed (16 keys with
    # preprocessing_date + config). Provenance (variant/cohort) lives inside `config`
    # and the sidecar manifest.
    data = {
        "exp_epochs": exp["epochs"],
        "ctrl_epochs": ctrl["epochs"],
        "exp_subjects": exp["subjects"],
        "ctrl_subjects": ctrl["subjects"],
        "exp_quality": exp["quality"],
        "ctrl_quality": ctrl["quality"],
        "clinical_data": clinical_data,
        "clinical_scores": clinical_scores,
        "demographics": demographics,
        "responses": responses,
        "quality_report": quality_report,
        "channel_strategy": channel_strategy,
        "epoch_rejection_enabled": epoch_rejection_config.get("enabled", True),
        "channel_masks": {**exp["channel_masks"], **ctrl["channel_masks"]},
        "preprocessing_date": datetime.now().isoformat(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    # One directory per variant: processed/dataset/eeg_only/<variant>/
    # holding epochs.pkl + manifest.json.
    out_dir = os.path.join(base, cfg.paths.output_dir, variant)
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, "epochs.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    # Date-stamped copy for the ANALYSIS repo to consume. The analysis side
    # should only ever read a finished pickle -- it holds no processing code --
    # so it gets a stable, self-describing filename rather than having to know
    # this repo's variant-directory convention. See docs/dataset_handoff.md.
    dated_name = f"synapse_preprocessed_{datetime.now():%Y-%m-%d}.pkl"
    dated_path = os.path.join(out_dir, dated_name)
    with open(dated_path, "wb") as f:
        pickle.dump(data, f)

    manifest = {
        "product": "epochs",
        "variant": variant,
        "cohort": cfg.cohort.name,
        "built": data["preprocessing_date"],
        "git_sha": qpaths.git_sha(),
        "channel_strategy": channel_strategy,
        "preprocessing": OmegaConf.to_container(pre, resolve=True),
        "tasks": task_timings,
        "exp_subjects": exp["subjects"],
        "ctrl_subjects": ctrl["subjects"],
        "epochs_per_task": {
            "EXP": {t: len(exp["epochs"][t]) for t in task_order},
            "CTRL": {t: len(ctrl["epochs"][t]) for t in task_order},
        },
        "resolved": resolved,
        "skipped": skipped,
        "clinical_scores_nonempty": sum(1 for v in clinical_scores.values() if v),
        "demographics_nonempty": len(demographics),
        "responses_loaded": len(responses),
        "pkl": os.path.relpath(pkl_path, base),
        "analysis_pkl": os.path.relpath(dated_path, base),
        "pkl_mb": round(os.path.getsize(pkl_path) / 1024 / 1024, 1),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Wrote {pkl_path}  ({manifest['pkl_mb']} MB)")
    print(f"      {dated_path}  (for the analysis repo)")
    print(f"  EXP {len(exp['subjects'])} / CTRL {len(ctrl['subjects'])} subjects")
    print(f"  epochs/task EXP={manifest['epochs_per_task']['EXP']} "
          f"CTRL={manifest['epochs_per_task']['CTRL']}")
    if skipped:
        print(f"  skipped: {[s['pid'] for s in skipped]}")


if __name__ == "__main__":
    main()
