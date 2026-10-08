#!/usr/bin/env python3
"""Sync the study's upstream sources, then score any new recordings.

One command for the recurring "is there new data?" chore. Stages run in order
because each depends on the one before it (numbering matches the table in
``docs/data_sync.md``):

1. **Box -> repo workbook.** The coordinator maintains the clinical workbook on
   Box; ``processing/02_PCData.xlsx`` is the repo's copy and goes stale. Pull it
   first so everything downstream judges against current data. Add
   ``--commit-workbook`` to also git-commit it (that one file, nothing else).
2. **Workbook -> Drive.** Mirror the same file into the Drive ``PC/`` folder so
   recordings and labels live together (the coordinator only ever sees Drive).
3. **Drive -> local recordings.** Diff ``PC/{Control,Experimental}`` against the
   local ``raw/`` tree and download whatever is missing. When the raw tree IS
   the Drive mount (``$SYNAPSE_DATA_BASE`` = the mounted ``PC/``) there is
   nothing to download; "new" then means not yet in the QC workbook.
4. **Demographics gate.** A recording is only scored once its demographics exist
   in the workbook, since a subject with no clinical row cannot enter a dataset
   anyway. ``--skip-demographics-check`` scores it regardless, for when you want
   the QC numbers before the coordinator has caught up on data entry.
5. **QC.** ``run_quality.py`` scores every recording.
6. **Publish the QC workbook** to Drive ``PC/quality_results.xlsx`` -- a single
   always-current file, deliberately not dated.
7. **Refresh built variants' ``labels.csv``.** A workbook pull changes the
   LABELS of an already-built dataset and nothing else, so rewrite just that file
   per variant rather than re-running finalize over ~360 MB of unchanged epochs.
   Runs whenever the workbook moved, independently of whether QC ran.

Step 8 -- actually building a dataset from new recordings (``pair_video`` +
``finalize_dataset``) -- stays MANUAL: it needs a cohort decision first, and
pairing is slow and irreversible.

Stages 1-2 always run. ``--clinical-only`` runs 1, 2 and 7 -- the fast path for
"the coordinator entered new demographics, update the datasets" with no download
and no QC.

Box always goes through the pre-configured ``box:`` rclone remote. Drive goes
through the ``gdrive:`` remote by default (the lab server), or -- with
``--drive-mount`` / ``$SYNAPSE_DRIVE_MOUNT`` -- through a Google Drive for
Desktop mount of the upload root (a workstation). Neither source folder is in
the account's own root, so every rclone path MUST be rooted by folder id, and a
mount is checked against that same id -- see ``BOX_FOLDER_ID`` /
``DRIVE_ROOT_ID`` below.

**``docs/data_sync.md`` is the reference for all of this** -- the by-hand
commands, the non-EEG sibling folders, the phone-only subject ids that look like
missing controls, and the workbook's shifting column layout.

    python -m scripts.sync_study_data --dry-run      # report only, change nothing
    python -m scripts.sync_study_data                # full sync + QC + labels
    python -m scripts.sync_study_data --clinical-only        # labels only
    python -m scripts.sync_study_data --clinical-only --variant usable__zero_mask
    python -m scripts.sync_study_data --skip-demographics-check
    python -m scripts.sync_study_data --no-qc        # sync only
    python -m scripts.sync_study_data --dry-run \\
        --drive-mount ~/"Google Drive/My Drive/PC-Synapse AudioSight/Participant Data"
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Box: 'AudioSight Study/01_Participant Data'. Addressed by id, not path, so a
# folder rename upstream does not break the pull.
BOX_FOLDER_ID = "324186906035"
BOX_WORKBOOK = "02_PC Data.xlsx"          # NB: space in the Box filename

# Drive: the coordinator's upload root (owned by lizr89721@gmail.com).
DRIVE_ROOT_ID = "1Eh9SlATsEUzrGEDgNVfWPLeOWE2Cy90l"
DRIVE_PC = "PC"                            # PC/ = EEG; Phone/, DNQ/ are not

# Drive subfolder -> local data subdir. The names deliberately differ.
COHORTS = {"Control": "control", "Experimental": "experimental"}

LOCAL_WORKBOOK = REPO / "02_PCData.xlsx"   # no space locally

# macOS uploads litter every folder with these; they break folder discovery.
JUNK = ["._*", ".DS_Store", ".*/._*"]

# A subject needs these before QC is worth running (see --skip-demographics-check).
REQUIRED_CLINICAL = ("age", "sex")


def sh(args, check=True, quiet=False):
    """Run a command, returning stdout. rclone's shared-client_id notice goes to
    stderr on every call, so stderr is captured and only shown on failure."""
    if not quiet:
        print(f"  $ {' '.join(args)}")
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(args)}\n"
                           f"{r.stderr.strip()}")
    return r.stdout


def rclone(*args, root_flag=None, root_id=None, **kw):
    cmd = ["rclone", *args]
    if root_flag:
        cmd += [root_flag, root_id]
    return sh(cmd, **kw)


def drive(*args, **kw):
    return rclone(*args, root_flag="--drive-root-folder-id",
                  root_id=DRIVE_ROOT_ID, **kw)


def box(*args, **kw):
    return rclone(*args, root_flag="--box-root-folder-id",
                  root_id=BOX_FOLDER_ID, **kw)


def _junk(name):
    return any(fnmatch.fnmatch(name, j) for j in JUNK)


def _drivefs_item_id(path):
    """The Drive id that Google Drive for Desktop records on a mounted folder,
    or None where it is not available (another sync client, not macOS)."""
    try:
        r = subprocess.run(["xattr", "-p", "com.google.drivefs.item-id#S",
                            str(path)], capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


# --------------------------------------------------------------------------- #
# Drive backends: the gdrive: rclone remote, or a Drive for Desktop mount.
# Both take paths relative to the upload root DRIVE_ROOT_ID.
# --------------------------------------------------------------------------- #
class RcloneDrive:
    """Drive through the ``gdrive:`` rclone remote -- the lab-server default."""

    def __str__(self):
        return f"rclone gdrive: (root {DRIVE_ROOT_ID})"

    def list_dirs(self, rel):
        out = drive("lsf", "--dirs-only", f"gdrive:{rel}", quiet=True)
        return sorted(x.rstrip("/") for x in out.split() if x.strip("/"))

    def has_xdf(self, rel):
        out = drive("lsf", "--recursive", "--files-only", "--include", "*.xdf",
                    f"gdrive:{rel}", quiet=True)
        return bool(out.strip())

    def size(self, rel):
        out = drive("size", f"gdrive:{rel}", quiet=True)
        return out.strip().splitlines()[-1].strip()

    def download(self, rel, dest):
        args = ["copy", f"gdrive:{rel}", str(dest)]
        for j in JUNK:
            args += ["--exclude", j]
        drive(*args, "-P")

    def upload(self, src, rel):
        # copyto compares size+mtime and skips a genuinely identical file.
        drive("copyto", str(src), f"gdrive:{rel}")
        return drive("lsl", f"gdrive:{rel}", quiet=True).strip()


class MountDrive:
    """Drive through a Google Drive for Desktop mount -- no ``gdrive:`` remote.

    ``root`` is the local folder that IS the Drive folder ``DRIVE_ROOT_ID``
    ('Participant Data'). A path is only as good as the folder it resolves to,
    so the id Drive for Desktop stores on the folder is checked, not trusted.

    Two differences from rclone: reading a streamed file is what makes Drive
    for Desktop fetch it, so a download runs at network speed; and a write only
    QUEUES the upload -- it reaches Drive once the client has synced it.
    """

    def __init__(self, root):
        self.root = Path(root).expanduser()
        if not (self.root / DRIVE_PC).is_dir():
            raise RuntimeError(f"Drive mount {self.root} has no {DRIVE_PC}/ "
                               f"folder -- point it at Drive folder {DRIVE_ROOT_ID}")
        found = _drivefs_item_id(self.root)
        if found is None:
            print(f"WARNING: cannot read the Drive id of {self.root}; "
                  f"trusting the path")
        elif found != DRIVE_ROOT_ID:
            raise RuntimeError(f"Drive mount {self.root} is Drive folder {found}, "
                               f"not the upload root {DRIVE_ROOT_ID}")

    def __str__(self):
        return f"mount {self.root}"

    def _files(self, rel):
        for dirpath, dirnames, filenames in os.walk(self.root / rel):
            dirnames[:] = [d for d in dirnames if not _junk(d)]
            for f in filenames:
                if not _junk(f):
                    yield Path(dirpath) / f

    def list_dirs(self, rel):
        return sorted(p.name for p in (self.root / rel).iterdir()
                      if p.is_dir() and not p.name.startswith("."))

    def has_xdf(self, rel):
        return any(p.suffix == ".xdf" for p in self._files(rel))

    def size(self, rel):
        files = list(self._files(rel))
        total = sum(p.stat().st_size for p in files)
        return f"{len(files)} files, {total / 1e9:.2f} GB"

    def download(self, rel, dest):
        src = self.root / rel

        def copy(s, d):
            print(f"    {Path(s).relative_to(src)} "
                  f"({os.path.getsize(s) / 1e6:.0f} MB)", flush=True)
            return shutil.copy2(s, d)

        # A half-copied subject folder would look present to the next run's
        # diff and never be re-fetched, so an interrupted copy removes it.
        try:
            shutil.copytree(src, dest, copy_function=copy,
                            ignore=shutil.ignore_patterns(*JUNK))
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise

    def upload(self, src, rel):
        dest = self.root / rel
        if dest.exists() and sha(dest) == sha(src):
            return f"{dest} (unchanged)"
        # Drive for Desktop turns an overwrite into a NEW Drive file (new id)
        # and trashes the old one (seen 2026-10-08 on 02_PC Data.xlsx), so the
        # previous version survives only in the trash, ~30 days, not as a
        # revision. rclone's gdrive: copyto updates in place instead.
        shutil.copy2(src, dest)
        return (f"{dest} ({dest.stat().st_size} bytes; queued -- Drive for "
                f"Desktop uploads it in the background)")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_STEP = [0]


def header(title):
    """Numbered stage banner. Auto-increments so skipped stages don't leave
    gaps and the summary can't print out of order (it used to)."""
    _STEP[0] += 1
    print(f"\n{'=' * 72}\n{_STEP[0]}. {title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #
# 1. Box -> local workbook
# --------------------------------------------------------------------------- #
def pull_workbook(dry_run, stage_dir):
    """Returns (changed, note, path_to_current_workbook).

    Compares by content hash, not mtime: rclone rewrites mtime on every copy, so
    a hash is the only honest "did it change". The third return value is the
    workbook the LATER STAGES must read -- under --dry-run the local file is
    deliberately left untouched, so the demographics gate has to judge against
    the staged Box copy or it reports stale gaps that upstream already filled.
    """
    header(f"Box -> {LOCAL_WORKBOOK.name}")
    listing = box("lsl", f"box:{BOX_WORKBOOK}", quiet=True).strip()
    if not listing:
        raise RuntimeError(f"{BOX_WORKBOOK!r} not found in Box folder "
                           f"{BOX_FOLDER_ID} -- was it renamed?")
    print(f"  Box:   {listing}")
    before = sha(LOCAL_WORKBOOK) if LOCAL_WORKBOOK.exists() else None
    print(f"  local: {'absent' if before is None else LOCAL_WORKBOOK.stat().st_size} bytes")

    staged = Path(stage_dir) / "workbook.xlsx"
    box("copyto", f"box:{BOX_WORKBOOK}", str(staged))
    if sha(staged) == before:
        print("  -> unchanged")
        return False, "unchanged", LOCAL_WORKBOOK
    if dry_run:
        print("  -> WOULD UPDATE (dry-run; later stages read the staged copy)")
        return True, "would update", staged
    shutil.copy2(staged, LOCAL_WORKBOOK)
    print(f"  -> UPDATED ({LOCAL_WORKBOOK.stat().st_size} bytes)")
    return True, "updated", LOCAL_WORKBOOK


# --------------------------------------------------------------------------- #
# 2. workbook -> Drive PC/
# --------------------------------------------------------------------------- #
def push_workbook(dry_run, workbook, drv):
    header(f"{Path(workbook).name} -> Drive {DRIVE_PC}/{BOX_WORKBOOK}")
    dest = f"{DRIVE_PC}/{BOX_WORKBOOK}"
    if dry_run:
        print(f"  -> WOULD UPLOAD to {dest} via {drv} (dry-run)")
        return
    print(f"  -> uploaded: {drv.upload(workbook, dest)}")


# --------------------------------------------------------------------------- #
# git: commit the refreshed workbook
# --------------------------------------------------------------------------- #
def commit_workbook(dry_run):
    """Commit ONLY the workbook, so a sync never sweeps up unrelated work in
    progress. Opt-in (``--commit-workbook``) because committing is not this
    script's job by default."""
    header(f"git commit {LOCAL_WORKBOOK.name}")
    rel = LOCAL_WORKBOOK.relative_to(REPO.parent)
    status = sh(["git", "-C", str(REPO.parent), "status", "--porcelain", "--",
                 str(rel)], quiet=True).strip()
    if not status:
        print("  -> nothing to commit (workbook unchanged in git)")
        return
    branch = sh(["git", "-C", str(REPO.parent), "rev-parse",
                 "--abbrev-ref", "HEAD"], quiet=True).strip()
    print(f"  branch: {branch}   status: {status}")
    if dry_run:
        print("  -> WOULD COMMIT (dry-run)")
        return
    sh(["git", "-C", str(REPO.parent), "commit", "-m",
        "processing: refresh the clinical workbook from Box", "--", str(rel)])
    print("  -> committed")


# --------------------------------------------------------------------------- #
# Drive PC/ -> local data/
# --------------------------------------------------------------------------- #
def sync_recordings(data_root, dry_run, drv):
    header(f"Drive {DRIVE_PC}/ -> {data_root}")
    new = {}
    for cohort, local_name in COHORTS.items():
        local_dir = Path(data_root) / local_name
        remote = drv.list_dirs(f"{DRIVE_PC}/{cohort}")
        local = sorted(p.name for p in local_dir.iterdir() if p.is_dir()) \
            if local_dir.exists() else []
        missing = [p for p in remote if p not in local]
        extra = [p for p in local if p not in remote]
        print(f"  {cohort:12s} drive={len(remote):3d}  local={len(local):3d}  "
              f"missing={missing or '-'}  local-only={extra or '-'}")
        for pid in missing:
            # A folder without an .xdf is not an EEG session (audiometry-only,
            # or a phone-era upload) -- do not pull it into the EEG tree.
            if not drv.has_xdf(f"{DRIVE_PC}/{cohort}/{pid}"):
                print(f"    {pid}: SKIPPED — no .xdf (not an EEG session)")
                continue
            new[pid] = (cohort, local_dir / pid)
    if not new:
        print("  -> nothing to download")
        return {}
    for pid, (cohort, dest) in new.items():
        src = f"{DRIVE_PC}/{cohort}/{pid}"
        print(f"  {pid}: {drv.size(src)}")
        if dry_run:
            print(f"    -> WOULD DOWNLOAD to {dest} (dry-run)")
            continue
        drv.download(src, dest)
        print(f"    -> {dest}")
    return new


def unscored_recordings(data_root):
    """Stage 3 when the raw tree IS the Drive mount (``$SYNAPSE_DATA_BASE`` =
    the mounted ``PC/``): nothing to download, so "new" means recorded but
    absent from the current QC workbook instead."""
    header(f"Drive {DRIVE_PC}/ is the raw tree -> recordings not yet scored")
    from synapse_qc import inventory
    from synapse_qc import paths as qpaths
    xlsx = Path(qpaths.output_paths()["qc"]) / "quality_results.xlsx"
    scored = set()
    if xlsx.exists():
        import pandas as pd
        scored = set(pd.read_excel(xlsx, sheet_name="Summary")["participant"])
    print(f"  QC workbook: {xlsx if xlsx.exists() else 'absent'} "
          f"({len(scored)} scored)")
    new = {}
    for p in inventory.discover(data_root=data_root):
        if p.pid in scored:
            continue
        if not p.xdf_path:
            print(f"    {p.pid}: SKIPPED — no .xdf (not an EEG session)")
            continue
        new[p.pid] = (p.group, Path(p.folder))
    print(f"  -> unscored: {sorted(new) or 'none'}")
    return new


def _raw_is_mount(data_root, drv):
    """True when the raw tree is the mounted Drive ``PC/`` folder itself --
    then a "download" would copy Drive into itself."""
    return (isinstance(drv, MountDrive) and os.path.isdir(data_root)
            and os.path.samefile(data_root, drv.root / DRIVE_PC))


# --------------------------------------------------------------------------- #
# 4. demographics gate
# --------------------------------------------------------------------------- #
def check_demographics(pids, workbook):
    """Returns {pid: [missing fields]} for subjects lacking clinical data."""
    header(f"Demographics gate (vs {workbook})")
    from synapse_qc.clinical import load_clinical_rows
    rows = {r["subject_id"]: r
            for r in load_clinical_rows(str(workbook), sorted(pids))}
    gaps = {}
    for pid in sorted(pids):
        r = rows.get(pid, {})
        missing = [f for f in REQUIRED_CLINICAL if not str(r.get(f, "")).strip()]
        n = sum(1 for k, v in r.items()
                if k not in ("subject_id", "group", "devices_present")
                and str(v).strip())
        flag = "OK " if not missing else "GAP"
        print(f"  {flag} {pid}: {n:3d} clinical fields"
              + (f" | MISSING {missing}" if missing else ""))
        if missing:
            gaps[pid] = missing
    return gaps


# --------------------------------------------------------------------------- #
# 5. QC + upload
# --------------------------------------------------------------------------- #
def run_qc(date, data_root, dry_run):
    header("Quality check")
    cmd = [sys.executable, "run_quality.py"]
    if date:
        cmd += ["--date", date]
    if data_root:
        cmd += ["--data-root", str(data_root)]
    if dry_run:
        print(f"  -> WOULD RUN: {' '.join(cmd)} (dry-run)")
        return None
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"run_quality.py failed ({r.returncode})")
    from synapse_qc import paths as qpaths
    xlsx = Path(qpaths.output_paths()["qc"]) / "quality_results.xlsx"
    if not xlsx.exists():
        raise RuntimeError(f"expected QC workbook not found: {xlsx}")
    return xlsx


def push_qc(xlsx, dry_run, drv):
    header(f"Quality workbook -> Drive {DRIVE_PC}/")
    dest = f"{DRIVE_PC}/quality_results.xlsx"
    if dry_run or xlsx is None:
        print(f"  -> WOULD UPLOAD {xlsx} to {dest} via {drv} (dry-run)")
        return
    print(f"  -> uploaded: {drv.upload(xlsx, dest)}")


# --------------------------------------------------------------------------- #
# Refresh labels.csv in the built dataset variants
# --------------------------------------------------------------------------- #
def refresh_clinical(workbook, dry_run, only_variant=None):
    """Re-derive each built variant's ``labels.csv`` from the current workbook.

    A workbook pull changes the LABELS of an already-built dataset, and nothing
    else -- the epochs and video are untouched -- so this rewrites just that one
    file per variant instead of re-running finalize (which would rebuild ~360 MB
    of .fif per variant for no reason). Mirrors ``finalize_dataset._write_clinical``:
    same column order (first-seen across rows) and the same manifest bookkeeping.
    """
    header("Refresh labels.csv in built variants")
    from omegaconf import OmegaConf
    from synapse_qc import paths as qpaths
    from synapse_qc.clinical import load_clinical_rows

    # Scan every dataset PRODUCT (eeg_only, multimodal, ...), not just the one
    # finalize happens to default to -- a workbook pull relabels all of them.
    base = qpaths.output_paths()["base"]
    fcfg = OmegaConf.load(REPO / "conf" / "finalize.yaml")
    final_dir = Path(base) / Path(fcfg.paths.dataset_dir).parent
    if not final_dir.exists():
        print(f"  -> no built variants at {final_dir}")
        return []
    variants = sorted(d for d in final_dir.glob("*/*")
                      if d.is_dir() and (d / "build_log.csv").exists())
    if only_variant:
        variants = [d for d in variants
                    if only_variant in (d.name, f"{d.parent.name}/{d.name}")]
        if not variants:
            raise RuntimeError(f"variant {only_variant!r} not found in {final_dir}")
    if not variants:
        print(f"  -> no finalized variants under {final_dir}")
        return []

    import csv as _csv
    import json
    import pandas as pd

    touched = []
    for v in variants:
        st = pd.read_csv(v / "build_log.csv")
        subjects = st.loc[st.status == "ok", "subject_id"].tolist()
        measures = None
        mf = v / "manifest.json"
        man = json.loads(mf.read_text()) if mf.exists() else {}
        if isinstance(man.get("clinical"), dict):
            measures = man["clinical"].get("measures")
        rows = load_clinical_rows(str(workbook), subjects, measures)
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        n_quest = sum(1 for r in rows if any(
            str(r.get(m, "")).strip() for m in (measures or [])))
        n_any = sum(1 for r in rows if any(
            str(val).strip() for k, val in r.items()
            if k not in ("subject_id", "group", "devices_present")))
        print(f"  {v.parent.name}/{v.name}: {len(rows)} subjects, {len(cols)} cols, "
              f"{n_any} with clinical data"
              + (f", {n_quest} with questionnaires" if measures else ""))
        if dry_run:
            print("    -> WOULD REWRITE labels.csv (dry-run)")
            continue
        with open(v / "labels.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        if man:
            man.setdefault("clinical", {})
            man["clinical"].update({
                "csv": "labels.csv",
                "workbook": str(workbook),
                "subjects_with_entries": n_any,
                "subjects_with_questionnaires": n_quest,
                "refreshed": datetime.date.today().isoformat(),
            })
            mf.write_text(json.dumps(man, indent=2))
        print("    -> rewrote labels.csv" + (" + manifest" if man else ""))
        touched.append(f"{v.parent.name}/{v.name}")
    return touched


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report every stage's decision, write nothing")
    ap.add_argument("--skip-demographics-check", action="store_true",
                    help="run QC even if new subjects have no clinical data yet")
    ap.add_argument("--no-qc", action="store_true",
                    help="stop after syncing; do not score or upload")
    ap.add_argument("--force-qc", action="store_true",
                    help="run QC even when no new recordings were downloaded")
    ap.add_argument("--data-root", default=None,
                    help="raw recordings dir (default: $SYNAPSE_DATA_ROOT / "
                         "$SYNAPSE_DATA_BASE/raw, or $SYNAPSE_DATA_BASE itself "
                         "when it is the Drive PC/ folder / <repo>/raw)")
    ap.add_argument("--date", default="", help="run-date label for the QC Legend")
    ap.add_argument("--clinical-only", action="store_true",
                    help="pull the workbook and refresh each built variant's "
                         "labels.csv, nothing else (no download, no QC)")
    ap.add_argument("--variant", default=None,
                    help="restrict the labels.csv refresh to one variant dir")
    ap.add_argument("--no-clinical-refresh", action="store_true",
                    help="leave built variants' labels.csv alone")
    ap.add_argument("--commit-workbook", action="store_true",
                    help="git-commit the refreshed workbook (that file only)")
    ap.add_argument("--drive-mount", default=os.environ.get("SYNAPSE_DRIVE_MOUNT"),
                    help="Google Drive for Desktop folder that is the upload root "
                         f"({DRIVE_ROOT_ID}); every Drive read/write goes through "
                         "it instead of rclone's gdrive: remote "
                         "(default: $SYNAPSE_DRIVE_MOUNT)")
    args = ap.parse_args()

    data_root = args.data_root
    if data_root is None:
        from synapse_qc import inventory
        data_root = inventory._default_data_root()
    drv = MountDrive(args.drive_mount) if args.drive_mount else RcloneDrive()
    print(f"data root: {data_root}")
    print(f"drive:     {drv}")
    if args.dry_run:
        print("DRY RUN — nothing will be written, uploaded or downloaded.")

    stage = tempfile.mkdtemp(prefix="synapse_sync_")
    try:
        # Stages 1-2 always run: the workbook is the cheapest thing to keep
        # current and everything downstream judges against it.
        wb_changed, wb_note, workbook = pull_workbook(args.dry_run, stage)
        push_workbook(args.dry_run, workbook, drv)
        if args.commit_workbook:
            commit_workbook(args.dry_run)

        if args.clinical_only:
            refreshed = refresh_clinical(workbook, args.dry_run, args.variant)
            header("Summary")
            print(f"  workbook:        {wb_note}")
            print(f"  labels.csv:    {refreshed or 'none'}")
            print("  recordings/QC:   skipped (--clinical-only)")
            print("\nDone.")
            return 0

        new = (unscored_recordings(data_root) if _raw_is_mount(data_root, drv)
               else sync_recordings(data_root, args.dry_run, drv))
        gaps = check_demographics(new, workbook) if new else {}

        header("Summary")
        print(f"  workbook:        {wb_note}")
        print(f"  new recordings:  {sorted(new) or 'none'}")
        print(f"  clinical gaps:   {gaps or 'none'}")

        rc, do_qc = 0, True
        if args.no_qc:
            print("  QC:              skipped (--no-qc)")
            do_qc = False
        elif not new and not args.force_qc:
            print("  QC:              skipped (no new recordings; "
                  "--force-qc to run)")
            do_qc = False
        elif gaps and not args.skip_demographics_check:
            print(f"  QC:              BLOCKED — {sorted(gaps)} lack "
                  f"{REQUIRED_CLINICAL} in the workbook.\n"
                  "                   The coordinator has not entered them yet. "
                  "Re-run later, or pass\n"
                  "                   --skip-demographics-check to score anyway.")
            do_qc, rc = False, 1

        if do_qc:
            xlsx = run_qc(args.date, args.data_root, args.dry_run)
            push_qc(xlsx, args.dry_run, drv)

        # A workbook pull changes the LABELS of already-built datasets, so
        # refresh them even when QC was skipped or blocked -- the two are
        # independent. Only worth doing if the workbook actually moved.
        if wb_changed and not args.no_clinical_refresh:
            refresh_clinical(workbook, args.dry_run, args.variant)
        elif not wb_changed:
            print("\n(labels.csv refresh skipped — workbook unchanged)")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print("\nDone." if rc == 0 else "\nStopped.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
