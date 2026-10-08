"""Discover participants and resolve which recording to use for each.

This is the shared data-layout layer: every pipeline (QC, and later the
application-specific subset pipelines) should resolve files through here so
the messy folder conventions live in exactly one place.

Folder layout (under the data base)::

    raw/experimental/EXP01/sub-323706/sub-323706_task-hearing_run-001.xdf
    raw/control/CTRL07/sub-184632/sub-184632_task-hearing_run-001.xdf

or, when the base is the Drive ``PC/`` folder, the cohort folders sit directly
in it (capitalized, next to ``processed/``)::

    Experimental/EXP01/sub-323706/...      Control/CTRL07/sub-184632/...

Non-obvious conventions handled here (verified against the actual data, not
assumed):
  * The workbook's Demographics ``New PC ID`` names a participant's CURRENT
    session. Normally the superseded one is renamed ``-old`` on Drive
    (EXP01/03/04), so the ranking finds it unaided; where it was not, the
    session is pinned in :data:`PREFERRED_SESSION` (CTRL21).
  * ``-old`` / ``_old`` sub-folders are re-done/abandoned sessions. Skip them
    when a current folder exists -- BUT CTRL01/02/03 have *only* an ``-old``
    folder, so it is used as a last resort (flagged ``old_only``).
  * EXP10 has a ``sub-... (Neurable)`` folder (a different 12-channel headset)
    next to the real 16-channel OpenBCI recording. The OpenBCI folder is
    preferred; Neurable is only used if it is the only option.
  * A participant may have an XDF but no marker stream, or no EEG stream at all
    (EXP26, CTRL03 = "no EEG data stream"). Resolution still returns the path;
    whether the EEG stream loads is determined downstream by the QC driver.
"""
import os
import glob
from dataclasses import dataclass, field

_THIS = os.path.dirname(os.path.abspath(__file__))


GROUP_DIRS = {
    "EXP": "experimental",
    "CTRL": "control",
}


def _group_dir(root, grp):
    """``root``'s folder for group ``grp``, or None. Matched case-insensitively:
    the lab-server tree says ``control``, the Drive ``PC/`` folder ``Control``."""
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return None
    for name in names:
        if name.lower() == GROUP_DIRS[grp] and os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    return None


def raw_root(base):
    """The raw-recordings dir of a data base. ``<base>/raw`` normally;
    ``<base>/data`` if only that exists (the name before the 2026-09-04 layout
    migration); ``<base>`` itself if the cohort folders sit directly in it --
    the Drive ``PC/`` folder, where the coordinator's upload tree IS the raw
    tree and ``processed/`` lives beside it."""
    new, legacy = os.path.join(base, "raw"), os.path.join(base, "data")
    if not os.path.isdir(new):
        if os.path.isdir(legacy):
            return os.path.normpath(legacy)
        if any(_group_dir(base, g) for g in GROUP_DIRS):
            return os.path.normpath(base)
    return os.path.normpath(new)


def _default_data_root():
    """Where the raw tree lives, resolved at import time.

    Lets the recordings be relocated off the repo (the lab server
    ``/data1/anarghya/synapse-data``, or the Drive ``PC/`` mount) without
    editing code. Precedence:
      1. ``$SYNAPSE_DATA_ROOT`` -- the raw-data dir directly.
      2. ``$SYNAPSE_DATA_BASE`` -- a base holding the raw tree + ``processed/``;
         see :func:`raw_root` for where the raw tree is looked for.
      3. the repo dir, same rules.
    Hydra pipelines can still override per-call via ``discover(data_root=...)``.
    """
    env = os.environ.get("SYNAPSE_DATA_ROOT")
    if env:
        return os.path.normpath(env)
    return raw_root(os.environ.get("SYNAPSE_DATA_BASE")
                    or os.path.join(_THIS, ".."))


DATA_ROOT = _default_data_root()


@dataclass
class Participant:
    pid: str                 # e.g. "EXP01"
    group: str               # "EXP" or "CTRL"
    folder: str              # absolute path to the EXP01/ folder
    sub_dir: str = ""        # chosen sub-XXXXXX dir name ("" if none)
    sub_id: str = ""         # numeric sub id, e.g. "323706"
    xdf_path: str = ""       # absolute path to chosen .xdf ("" if none)
    responses_csv: str = ""  # absolute path to responses CSV ("" if none)
    video: str = ""          # absolute path to .avi ("" if none)
    pdfs: list = field(default_factory=list)   # audiogram / tymp PDFs
    old_only: bool = False   # only an -old session was available
    is_neurable: bool = False  # chosen FOLDER is name-tagged Neurable (heuristic; device is confirmed from the stream downstream)
    n_sub_dirs: int = 0      # how many sub-* folders exist (for auditing)
    all_xdfs: list = field(default_factory=list)  # every .xdf across all sub-folders (for device detection)
    note: str = ""           # resolution note (e.g. why a folder was picked)


# Sessions the workbook's 'New PC ID' designates as current, for participants
# whose superseded session was NOT renamed '-old' (so the ranking alone would
# pick it). Keep this in step with Demographics 'New PC ID'.
PREFERRED_SESSION = {
    "CTRL21": "sub-800412",   # re-test 2026-09-18; first session sub-307951
}


def _is_old(name):
    n = name.lower()
    return "-old" in n or "_old" in n


def _is_neurable(name):
    return "neurable" in name.lower()


def _first_xdf(sub_path):
    files = sorted(glob.glob(os.path.join(sub_path, "*.xdf")))
    return files[0] if files else ""


def _first(sub_path, pattern):
    files = sorted(glob.glob(os.path.join(sub_path, pattern)))
    return files[0] if files else ""


def resolve_participant(pid, group, folder):
    """Choose the canonical recording for one participant folder.

    Preference order among sub-* folders that actually contain an XDF:
      current/OpenBCI  >  current/Neurable  >  old/OpenBCI  >  old/Neurable
    """
    sub_dirs = [d for d in sorted(os.listdir(folder))
                if d.startswith("sub-") and os.path.isdir(os.path.join(folder, d))]
    p = Participant(pid=pid, group=group, folder=folder, n_sub_dirs=len(sub_dirs))

    # Audiogram / tympanometry PDFs live at the participant level.
    p.pdfs = sorted(glob.glob(os.path.join(folder, "*.pdf")))

    # Rank candidate sub-folders that contain an XDF.
    candidates = []
    for d in sub_dirs:
        sub_path = os.path.join(folder, d)
        xdf = _first_xdf(sub_path)
        if not xdf:
            continue
        old = _is_old(d)
        neur = _is_neurable(d)
        pinned = d == PREFERRED_SESSION.get(pid)
        rank = (0 if pinned else 1, 0 if not old else 1,
                0 if not neur else 1)  # lower = better
        candidates.append((rank, d, sub_path, xdf, old, neur))

    p.all_xdfs = [c[3] for c in candidates]
    if not candidates:
        p.note = "no sub-folder with an XDF"
        return p

    candidates.sort(key=lambda c: c[0])
    _, d, sub_path, xdf, old, neur = candidates[0]
    p.sub_dir = d
    p.sub_id = d.replace("sub-", "").replace("-old", "").replace("_old", "").strip()
    p.xdf_path = xdf
    p.old_only = old
    p.is_neurable = neur
    p.responses_csv = _first(sub_path, "*responses*.csv") or _first(sub_path, "*.csv")
    p.video = _first(sub_path, "*.avi")

    notes = []
    if old:
        notes.append("only an -old session available")
    if neur:
        notes.append("Neurable headset (12ch)")
    if len(candidates) > 1:
        why = " (workbook New PC ID)" if d == PREFERRED_SESSION.get(pid) else ""
        notes.append(f"{len(candidates)} sessions; picked '{d}'{why}")
    p.note = "; ".join(notes)
    return p


def discover(data_root=None, groups=("EXP", "CTRL")):
    """Return a list of resolved :class:`Participant` for every folder found."""
    root = data_root or DATA_ROOT
    out = []
    for grp in groups:
        gdir = _group_dir(root, grp)
        if not gdir:
            continue
        for pid in sorted(os.listdir(gdir)):
            if not pid.startswith(grp):
                continue
            folder = os.path.join(gdir, pid)
            if not os.path.isdir(folder):
                continue
            out.append(resolve_participant(pid, grp, folder))
    return out


if __name__ == "__main__":
    # Quick audit of resolution decisions.
    for p in discover():
        flags = []
        if p.old_only:
            flags.append("OLD-ONLY")
        if p.is_neurable:
            flags.append("NEURABLE")
        if not p.xdf_path:
            flags.append("NO-XDF")
        print(f"{p.pid:8} {p.sub_dir:24} xdf={'Y' if p.xdf_path else 'N'} "
              f"csv={'Y' if p.responses_csv else 'N'} avi={'Y' if p.video else 'N'} "
              f"pdf={len(p.pdfs)} {' '.join(flags)}  {p.note}")
