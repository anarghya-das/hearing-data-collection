"""Per-subject clinical rows straight from the PC workbook (02_PCData.xlsx).

Self-contained replacement for the ``../../synapse`` published clinical loaders,
so ``clinical.csv`` can be built on any machine from the workbook alone. Parses:

* ``Demographics``  — one row per subject (ids, age/sex/race, dx history)
* the questionnaire sheets (THI/GAD/HQ/Iowa/Misophonia scores), in one of two
  layouts:

  - **per device** (the workbook since 2026-10-05): ``Questionnaires (OpenBCi)``
    and ``Questionnaires (Neurable)``, one row per subject per headset session.
    A subject's scores come from the OpenBCi sheet if it has a row there (every
    built dataset uses the OpenBCI EEG, so the questionnaire must be the one
    taken at that session — EXP10 has both), else from the Neurable sheet
    (Neurable-only subjects: EXP32/42/55). ``questionnaire_device`` records
    which. The Demographics ``PC ID`` / ``New PC ID`` / ``Neurable ID`` columns
    say which recording (``sub-<id>``) belongs to which headset.
  - **legacy** (older workbooks, e.g. the published one): a primary
    ``Questionnaires`` sheet, with ``Questionnaires (OpenBCi)`` only as a
    **fallback** for subjects absent from it (the later EXP batch: EXP20,
    EXP43-46, EXP50-54). Kept bit-identical so old builds reproduce.

  The device sheets carry condition-flag columns before THI, so every measure
  is shifted right relative to ``Questionnaires`` — hence a second position map.
* ``Audio Data``    — one wide row per subject (cohort flags, PTA + UHF
  thresholds, SRT, WRS, LDLs, tinnitus matching, tympanometry)
* ``Severity Data Sum`` — only as a tripwire: the scores we emit are checked
  against its hand-entered GAD category, which caught the OpenBCi sheet's
  CTRL06-09 rows sitting against the wrong IDs (see :func:`_check_gad_severity`).

The workbook uses multi-row decorated headers, so columns are addressed by
position and **validated against the header labels** — if the coordinator
restructures a sheet this fails loudly instead of silently misreading.
Field names mirror the published extractors (age, dx_neuro, pta_right_250, ...)
so downstream consumers see the same schema either way.
"""
import datetime
import math
import re

import pandas as pd

_ID_RE = re.compile(r"^(EXP|CTRL)\d+$")
_NUMERIC_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)$")

# Questionnaires sheet: measure -> column index (validated in _check_quest).
MEASURE_COLS = {
    "THI_Total": 3, "GAD_Total": 5,
    "HQ_Functional": 7, "HQ_Social": 8, "HQ_Emotional": 9, "HQ_Total": 10,
    "Iowa_Total": 15,
    "Miso_Section1": 17, "Miso_Section2": 18, "Miso_Section3": 19,
}

# The per-device sheets: same measures, shifted right by the condition-flag
# block (cols 2-9). Base positions; _quest_device_cols corrects for inserted or
# missing flag columns (the Neurable sheet has no COG column, the OpenBCi sheet
# gained TBI), so one map serves both sheets.
QUEST_SHEETS = {"OpenBCI": "Questionnaires (OpenBCi)",
                "Neurable": "Questionnaires (Neurable)"}
MEASURE_COLS_OPENBCI = {
    "THI_Total": 10, "GAD_Total": 12,
    "HQ_Functional": 14, "HQ_Social": 15, "HQ_Emotional": 16, "HQ_Total": 17,
    "Iowa_Total": 22,
    "Miso_Section1": 24, "Miso_Section2": 25, "Miso_Section3": 26,
}

_PTA_FREQS = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
_UHF_FREQS = [10000, 12500, 14000, 16000]
_LDL_FREQS = [250, 500, 1000, 2000, 4000, 8000]


def _cell(v):
    """Normalize a raw cell for CSV: NaN -> '', timestamps -> date, else as-is."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (pd.Timestamp, datetime.datetime)):
        return v.date().isoformat()
    if isinstance(v, datetime.date):
        return v.isoformat()
    return v


def _quest_cell(v):
    """Like :func:`_cell` but for questionnaire scores, which must stay numeric.

    The OpenBCi sheet writes the *text* ``"No Tinnitus"`` in THI_Total for every
    control (and EXP44/EXP53), where the primary Questionnaires sheet just leaves
    the cell blank. Normalize to blank so the two sheets agree and the column
    does not turn into an object dtype downstream.
    """
    v = _cell(v)
    if isinstance(v, str) and not _NUMERIC_RE.match(v.strip()):
        return ""
    return v


def _header_row(df, label="Group ID"):
    for i in range(min(len(df), 12)):
        if str(df.iat[i, 0]).strip() == label:
            return i
    raise ValueError(f"could not find header row (col0 == {label!r})")


def _expect(df, row, col, want, sheet):
    got = str(df.iat[row, col]).strip()
    ok = (got == want) if not want.endswith("*") else got.startswith(want[:-1])
    if not ok:
        raise ValueError(f"workbook layout changed: {sheet!r} header cell "
                         f"({row},{col}) is {got!r}, expected {want!r}")


def _find_label(df, row, want, lo, hi):
    """Column index in ``row`` whose cell equals ``want``, searched in [lo, hi).
    ``want`` may end in ``*`` for a prefix match. Returns None if not found."""
    for c in range(lo, min(hi, df.shape[1])):
        got = str(df.iat[row, c]).strip()
        if (got == want) if not want.endswith("*") else got.startswith(want[:-1]):
            return c
    return None


def _shift(df, row, anchor, expected, sheet):
    """How far the columns at/after ``expected`` have moved.

    The coordinator periodically inserts a new condition-flag column into these
    sheets (an ``ESL`` column once, a ``TBI`` column in the 2026-09-04 revision),
    which slides every measure to the right. Rather than re-hardcoding the
    positions each time, locate ``anchor`` near where it should be and return the
    delta to add to the tail columns. A shift of 0 means the layout is unchanged.
    """
    got = _find_label(df, row, anchor, max(0, expected - 4), expected + 6)
    if got is None:
        raise ValueError(f"workbook layout changed: {sheet!r} row {row} has no "
                         f"{anchor!r} column near {expected} — cannot resolve "
                         f"the column layout")
    return got - expected


def _subject_rows(df, id_col=0):
    """{PID: row_index} for every row whose id column is an EXP/CTRL id.
    First occurrence wins (the sheets can repeat ids in 'Not Sorted' blocks)."""
    out = {}
    for i in range(len(df)):
        pid = str(df.iat[i, id_col]).strip()
        if _ID_RE.match(pid) and pid not in out:
            out[pid] = i
    return out


def _check_quest(df):
    _expect(df, 1, 3, "THI", "Questionnaires")
    _expect(df, 1, 5, "GAD", "Questionnaires")
    _expect(df, 1, 7, "HQ", "Questionnaires")
    _expect(df, 1, 12, "Iowa*", "Questionnaires")
    _expect(df, 1, 17, "Miso*", "Questionnaires")
    _expect(df, 2, 10, "Total Score", "Questionnaires")


def _quest_device_cols(df, sheet):
    """``MEASURE_COLS_OPENBCI`` corrected for any inserted condition-flag
    columns, then validated. THI is the first measure after the flag block, so
    one anchor fixes every measure."""
    d = _shift(df, 1, "THI", MEASURE_COLS_OPENBCI["THI_Total"], sheet)
    cols = {m: c + d for m, c in MEASURE_COLS_OPENBCI.items()}
    _expect(df, 1, cols["GAD_Total"], "GAD", sheet)
    _expect(df, 1, cols["HQ_Functional"], "HQ", sheet)
    _expect(df, 2, cols["HQ_Total"], "Total Score", sheet)
    _expect(df, 2, cols["Iowa_Total"], "Total Score", sheet)
    _expect(df, 2, cols["Miso_Section1"], "Section 1", sheet)
    return cols


def _num(v):
    """``v`` as a float, or None for blanks / text / NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _gad_category(score):
    """GAD-7 total -> the category code 'Severity Data Sum' uses (its own
    legend: 1 = 5-7, 2 = 8-9, 3 = 10-14, 4 = 15-21; 0 below 5)."""
    return (0 if score < 5 else 1 if score < 8 else 2 if score < 10
            else 3 if score < 15 else 4)


def _severity_gad(df):
    """{PID: GAD category} from 'Severity Data Sum'. The sheet holds an EXP and
    a CTRL block, each headed by a ``Subject`` cell with ``GAD`` to its right."""
    out = {}
    for i in range(min(len(df), 12)):
        for j in range(df.shape[1]):
            if str(df.iat[i, j]).strip() != "Subject":
                continue
            g = _find_label(df, i, "GAD", j + 1, j + 8)
            if g is None:
                continue
            for r in range(i + 1, len(df)):
                pid, v = str(df.iat[r, j]).strip(), _num(df.iat[r, g])
                if _ID_RE.match(pid) and pid not in out and v in (0, 1, 2, 3, 4):
                    out[pid] = int(v)
    return out


def _check_gad_severity(gad, xls):
    """Raise if any emitted GAD total disagrees with the hand-entered GAD
    category in 'Severity Data Sum' -- the symptom of score rows pasted against
    the wrong IDs. It caught the OpenBCi sheet's CTRL06-09 rotation (CTRL06
    carried CTRL07's scores, ...); the deleted primary sheet agreed 34/34. It is
    a tripwire, not full validation: a swap within one GAD band (CTRL09) is
    invisible to it. ``gad`` is {PID: (score, sheet)}."""
    if "Severity Data Sum" not in xls.sheet_names:
        print("[clinical] WARNING: no 'Severity Data Sum' sheet -- cannot "
              "cross-check questionnaire rows against subject ids")
        return
    sev = _severity_gad(pd.read_excel(xls, sheet_name="Severity Data Sum",
                                      header=None))
    bad = []
    for sid, (raw, sheet) in gad.items():
        score = _num(raw)
        if score is not None and sid in sev and _gad_category(score) != sev[sid]:
            bad.append(f"{sid}: GAD {score:g} in {sheet!r} is category "
                       f"{_gad_category(score)}, 'Severity Data Sum' says {sev[sid]}")
    if bad:
        raise ValueError(
            "questionnaire scores disagree with 'Severity Data Sum' -- rows "
            "probably pasted against the wrong subject ids. Fix the workbook "
            "on Box; do not work around it in code (docs/data_sync.md):\n  "
            + "\n  ".join(bad))


# Audio Data: everything from this column onward slides when a condition flag
# is inserted; the columns before it (ids, dates, cohort flags) never move.
_AUDIO_TAIL = 22


def _audio_shift(df, hdr):
    """Columns >= :data:`_AUDIO_TAIL` shift when a condition flag is inserted.
    The first PTA frequency (250 Hz) is the anchor."""
    return _shift(df, hdr, "250", _AUDIO_TAIL, "Audio Data")


def _check_audio(df, hdr, d=0):
    for col, want in [(4, "Audio ID"), (5, "Date Tested (O*"),
                      (7, "Date Tested (N*"),
                      (15, "Control"), (19, "Misophonia"),
                      (30, "PTA"), (41, "PTA"), (44, "SRT RE"), (47, "WRS RE*"),
                      (68, "Total Score*"), (76, "Total Score*"),
                      (78, "Ear Tested"), (85, "Tymp RE"), (86, "Tymp LE"),
                      (88, "Average LDL*")]:
        _expect(df, hdr, col + (d if col >= _AUDIO_TAIL else 0), want,
                "Audio Data")
    for col, freq in [(22, 250), (33, 250), (52, 10000), (57, 10000),
                      (62, 250), (70, 250)]:
        if int(float(df.iat[hdr, col + d])) != freq:
            raise ValueError(f"workbook layout changed: 'Audio Data' header "
                             f"col {col + d} != {freq}")


def _demographics(df, hdr, i):
    r = df.iloc[i]
    return {
        "age": _cell(r[7]), "sex": _cell(r[8]), "race": _cell(r[9]),
        "date_tested": _cell(r[5]), "dob": _cell(r[6]),
        "dx_neuro": _cell(r[10]), "dx_migraine": _cell(r[11]),
        "dx_mental": _cell(r[12]), "dx_visual": _cell(r[13]),
        "light_sensitivity": _cell(r[14]), "neck_problem": _cell(r[15]),
        "tmj": _cell(r[16]),
        "pc_id": _cell(r[1]), "new_pc_id": _cell(r[2]),
        "neurable_id": _cell(r[3]), "audio_id": _cell(r[4]),
    }


def _audio(df, i, d=0):
    r = df.iloc[i]
    def t(col):
        """Tail column, corrected for inserted condition-flag columns."""
        return r[col + d]
    row = {
        "date_tested_openbci": _cell(r[5]), "date_tested_neurable": _cell(r[7]),
        "is_control": _cell(r[15]), "has_hearing_loss": _cell(r[16]),
        "has_tinnitus": _cell(r[17]), "has_hyperacusis": _cell(r[18]),
        "has_misophonia": _cell(r[19]),
    }
    for base, freqs in [(22, _PTA_FREQS), (52, _UHF_FREQS)]:
        for k, f in enumerate(freqs):
            row[f"pta_right_{f}"] = _cell(t(base + k))
    for base, freqs in [(33, _PTA_FREQS), (57, _UHF_FREQS)]:
        for k, f in enumerate(freqs):
            row[f"pta_left_{f}"] = _cell(t(base + k))
    row.update({
        "pta_right_avg": _cell(t(30)), "hearing_loss_right": _cell(t(31)),
        "pta_left_avg": _cell(t(41)), "hearing_loss_left": _cell(t(42)),
        "srt_right": _cell(t(44)), "srt_left": _cell(t(45)),
        "wrs_right_pct": _cell(t(47)), "wrs_right_db": _cell(t(48)),
        "wrs_left_pct": _cell(t(49)), "wrs_left_db": _cell(t(50)),
    })
    for k, f in enumerate(_LDL_FREQS):
        row[f"ldl_right_{f}"] = _cell(t(62 + k))
    row["ldl_right_avg"] = _cell(t(68))
    for k, f in enumerate(_LDL_FREQS):
        row[f"ldl_left_{f}"] = _cell(t(70 + k))
    row.update({
        "ldl_left_avg": _cell(t(76)), "ldl_overall_avg": _cell(t(88)),
        "tinnitus_ear_tested": _cell(t(78)), "tinnitus_noise_used": _cell(t(79)),
        "tinnitus_pitch_hz": _cell(t(80)), "tinnitus_loudness_db": _cell(t(81)),
        "tympanometry_right": _cell(t(85)), "tympanometry_left": _cell(t(86)),
    })
    return row


def load_clinical_rows(xlsx_path, subjects, measures=None):
    """One flat dict per subject (in the given order): questionnaire scores for
    ``measures`` (default: all of :data:`MEASURE_COLS`) + demographics +
    audiometry. Subjects absent from a sheet just get those fields empty."""
    measures = list(measures) if measures else list(MEASURE_COLS)
    unknown = [m for m in measures if m not in MEASURE_COLS]
    if unknown:
        raise ValueError(f"unknown clinical measures {unknown}; "
                         f"available: {sorted(MEASURE_COLS)}")

    xls = pd.ExcelFile(xlsx_path)
    # Legacy layout iff the primary sheet exists (see the module docstring).
    legacy = "Questionnaires" in xls.sheet_names
    if legacy:
        quest = pd.read_excel(xls, sheet_name="Questionnaires", header=None)
        _check_quest(quest)
        q_rows = _subject_rows(quest)
    # device -> (sheet name, frame, measure cols, {PID: row}). The legacy
    # layout uses only the OpenBCi sheet (as its fallback); the Neurable sheet
    # is optional either way, since older workbooks predate it.
    dev = {}
    for device, sheet in QUEST_SHEETS.items():
        if device == "Neurable" and (legacy or sheet not in xls.sheet_names):
            continue
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        dev[device] = (sheet, df, _quest_device_cols(df, sheet), _subject_rows(df))
    demo = pd.read_excel(xls, sheet_name="Demographics", header=None)
    audio = pd.read_excel(xls, sheet_name="Audio Data", header=None)

    demo_hdr = _header_row(demo)
    _expect(demo, demo_hdr, 8, "Biological Sex", "Demographics")
    audio_hdr = _header_row(audio)
    audio_d = _audio_shift(audio, audio_hdr)
    _check_audio(audio, audio_hdr, audio_d)

    d_rows, a_rows = _subject_rows(demo), _subject_rows(audio)

    out, gad = [], {}
    for sid in subjects:
        row = {"subject_id": sid,
               "group": "EXP" if sid.startswith("EXP") else "CTRL"}
        if legacy and sid in q_rows:
            # Primary sheet wins; the OpenBCi sheet only fills subjects it omits.
            for m in measures:
                row[m] = _quest_cell(quest.iat[q_rows[sid], MEASURE_COLS[m]])
            gad[sid] = (quest.iat[q_rows[sid], MEASURE_COLS["GAD_Total"]],
                        "Questionnaires")
        else:
            # The OpenBCi session's questionnaire if there is one, else the
            # Neurable session's (legacy: OpenBCi only).
            src = next((d for d in ("OpenBCI", "Neurable")
                        if d in dev and sid in dev[d][3]), None)
            if src:
                sheet, df, cols, rows = dev[src]
                for m in measures:
                    row[m] = _quest_cell(df.iat[rows[sid], cols[m]])
                gad[sid] = (df.iat[rows[sid], cols["GAD_Total"]], sheet)
            if not legacy:
                row["questionnaire_device"] = src or ""
        if sid in d_rows:
            row.update(_demographics(demo, demo_hdr, d_rows[sid]))
        if sid in a_rows:
            row.update(_audio(audio, a_rows[sid], audio_d))
        # Devices the workbook records for this subject (a subject can have
        # both, e.g. EXP10). NB: the EEG in the multimodal dataset is always
        # the OpenBCI recording — pairing requires the obci_eeg1 stream.
        devices = []
        if any(str(row.get(k, "")).strip()
               for k in ("pc_id", "new_pc_id", "date_tested_openbci")):
            devices.append("OpenBCI")
        if any(str(row.get(k, "")).strip()
               for k in ("neurable_id", "date_tested_neurable")):
            devices.append("Neurable")
        row = {"subject_id": sid, "group": row["group"],
               "devices_present": ", ".join(devices), **row}
        out.append(row)
    _check_gad_severity(gad, xls)
    return out
