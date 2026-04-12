#!/usr/bin/env python3
"""
Compare NB04 evaluation CSVs between two run tags (apples-to-apples by variant).

Relative change is always: (new - baseline) / baseline.

Interpretation:
  - Higher is better for AUC, F1, precision, recall, accuracy.
  - Lower is better for MAE; a positive rel_change on MAE means worse.

Output CSVs include rel_change (fraction) and rel_change_pct (e.g. +9.63% or n/a).

Never mix variants: compare best_prauc vs best_prauc and best_mcmae vs best_mcmae only.

When each variant was saved from a separate NB04 run, point PR-AUC and MCMAE folders
explicitly for *new* and *baseline* (any omitted flag falls back to --new-tag / --baseline-tag).

Typical layout (example tags from this repo):
  - best_prauc CSVs: .../20260412_082801/ (new), .../20260322_074843/ (baseline)
  - best_mcmae CSVs: .../20260412_081734/ (new), .../20260322_074438/ (baseline)

  python scripts/compare_eval_runs.py \\
    --runs-dir report_outputs/runs \\
    --new-tag 20260412_082801 \\
    --new-best-mcmae-tag 20260412_081734 \\
    --baseline-tag 20260322_074843 \\
    --baseline-best-mcmae-tag 20260322_074438 \\
    --variants best_prauc best_mcmae

Use --new-best-prauc-tag / --baseline-best-prauc-tag only if PR-AUC CSVs are not under
--new-tag / --baseline-tag.

Run from the repo root (or pass an absolute --runs-dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_VARIANTS = ("best_prauc", "best_mcmae")

# Per-label counts: still output baseline/new in long format, but omit rel_change (not meaningful as %).
SKIP_REL_CHANGE_COLS = frozenset({"support_pos", "support_neg"})

# Quick terminal summary (overall, test split only)
SUMMARY_OVERALL_METRICS = (
    "overall_PR_AUC",
    "overall_F1@tuned",
    "overall_F1@0.5",
    "score_mae",
    "veg_mae",
    "shade_acc_overall",
    "score_acc",
    "veg_acc",
)


def _resolve_runs_dir(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    cwd = Path.cwd()
    cand = (cwd / p).resolve()
    if cand.exists():
        return cand
    # Repo root: parent of scripts/ when invoked as scripts/compare_eval_runs.py
    here = Path(__file__).resolve().parent
    root = here.parent
    cand2 = (root / p).resolve()
    return cand2


def _load_eval_csv(path: Path) -> pd.DataFrame:
    """Load NB04 export; handle UTF-8 BOM, odd column names, and one junk title row.

    If someone prepends a title line (e.g. \"NEW BEST MAE\") before the real header,
    the first read has no 'split' column — retry with skiprows=1.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if "split" not in df.columns:
        df2 = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
        df2.columns = [str(c).strip().lstrip("\ufeff") for c in df2.columns]
        if "split" in df2.columns:
            return df2
    return df


def _validate_schema(
    path: Path,
    role: str,
    df: pd.DataFrame,
    required: list[str],
    variant: str,
) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(
            f"[{variant}] {role}\n  file: {path}\n"
            f"  missing columns: {missing}\n  found: {list(df.columns)}",
            file=sys.stderr,
        )
        return False
    if df.empty:
        print(f"[{variant}] {role}\n  file: {path}\n  empty CSV", file=sys.stderr)
        return False
    return True


def _discover_run_tags(runs_dir: Path) -> list[str]:
    if not runs_dir.is_dir():
        raise SystemExit(f"Runs directory does not exist: {runs_dir}")
    tags = sorted(
        (d.name for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")),
        reverse=True,
    )
    return tags


def _metric_columns(df: pd.DataFrame, merge_keys: list[str]) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if c in merge_keys:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _rel_change(baseline: float, new: float) -> float:
    if not (pd.notna(baseline) and pd.notna(new)):
        return float("nan")
    if baseline == 0:
        return float("nan")
    return (new - baseline) / baseline


def _format_rel_change_pct(rel: float) -> str:
    if not pd.notna(rel):
        return "n/a"
    sign = "+" if rel > 0 else ""
    return f"{sign}{rel * 100:.2f}%"


def _merge_long(
    df_new: pd.DataFrame,
    df_baseline: pd.DataFrame,
    merge_keys: list[str],
    variant: str,
    table: str,
    skip_rel_change: frozenset[str],
    new_tag_label: str,
    baseline_tag_label: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Inner merge; return long-format rows and notes for stderr."""
    notes: list[str] = []
    metrics = _metric_columns(df_new, merge_keys)
    for m in metrics:
        if m not in df_baseline.columns:
            notes.append(f"Metric {m!r} missing in baseline; skipped.")
    metrics = [m for m in metrics if m in df_baseline.columns]

    merged = pd.merge(
        df_new,
        df_baseline,
        on=merge_keys,
        how="inner",
        suffixes=("_new", "_baseline"),
    )

    rows: list[dict] = []

    for _, row in merged.iterrows():
        base_keys = {k: row[k] for k in merge_keys}
        for metric in metrics:
            col_n = f"{metric}_new"
            col_b = f"{metric}_baseline"
            if col_n not in row.index or col_b not in row.index:
                continue
            new_v = row[col_n]
            base_v = row[col_b]
            if metric in skip_rel_change:
                rel = float("nan")
            else:
                rel = _rel_change(float(base_v), float(new_v))
            r = {
                "variant": variant,
                "table": table,
                "metric": metric,
                "baseline": base_v,
                "new": new_v,
                "rel_change": rel,
                "rel_change_pct": _format_rel_change_pct(rel),
                "csv_new_tag": new_tag_label,
                "csv_baseline_tag": baseline_tag_label,
                **base_keys,
            }
            rows.append(r)

    out = pd.DataFrame(rows)
    return out, notes


def _label_set_diagnostic(
    df_new: pd.DataFrame,
    df_baseline: pd.DataFrame,
    variant: str,
) -> None:
    """Print labels present in only one run (per split)."""
    if "label" not in df_new.columns or "label" not in df_baseline.columns:
        return
    for split in ("train", "val", "test"):
        n = set(df_new.loc[df_new["split"] == split, "label"].astype(str))
        b = set(df_baseline.loc[df_baseline["split"] == split, "label"].astype(str))
        only_b = sorted(b - n)
        only_n = sorted(n - b)
        if only_b:
            print(f"[{variant}] Labels only in baseline ({split}): {only_b}", file=sys.stderr)
        if only_n:
            print(f"[{variant}] Labels only in new ({split}): {only_n}", file=sys.stderr)


def _print_summary(long_overall: pd.DataFrame, variant: str) -> None:
    sub = long_overall[
        (long_overall["split"] == "test") & (long_overall["metric"].isin(SUMMARY_OVERALL_METRICS))
    ]
    if sub.empty:
        return
    print(f"\n--- Test split summary ({variant}) ---")
    for metric in SUMMARY_OVERALL_METRICS:
        m = sub[sub["metric"] == metric]
        if m.empty:
            continue
        r = m.iloc[0]
        print(
            f"  {metric}: baseline={r['baseline']:.4f}  new={r['new']:.4f}  "
            f"rel_change_pct={r['rel_change_pct']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluation CSVs between two run tags (same variant only)."
    )
    parser.add_argument(
        "--runs-dir",
        default="report_outputs/runs",
        help="Directory containing <run_tag>/ subfolders (default: report_outputs/runs)",
    )
    parser.add_argument("--new-tag", default=None, help="Newer run tag (folder name); default root for new-side CSVs")
    parser.add_argument("--baseline-tag", default=None, help="Older run tag (folder name); default root for baseline CSVs")
    parser.add_argument(
        "--new-best-prauc-tag",
        default=None,
        help="Folder for new run best_prauc CSVs (default: --new-tag)",
    )
    parser.add_argument(
        "--new-best-mcmae-tag",
        default=None,
        help="Folder for new run best_mcmae CSVs (default: --new-tag)",
    )
    parser.add_argument(
        "--baseline-best-prauc-tag",
        default=None,
        help="Folder for baseline best_prauc CSVs (default: --baseline-tag)",
    )
    parser.add_argument(
        "--baseline-best-mcmae-tag",
        default=None,
        help="Folder for baseline best_mcmae CSVs (default: --baseline-tag)",
    )
    parser.add_argument(
        "--auto-last-two",
        action="store_true",
        help="Use the two newest run folders by name (descending sort)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help=f"Variants to compare (default: {' '.join(DEFAULT_VARIANTS)})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for CSVs (default: <runs-dir>/<new-tag>)",
    )
    args = parser.parse_args()

    runs_dir = _resolve_runs_dir(args.runs_dir)

    if args.auto_last_two:
        tags = _discover_run_tags(runs_dir)
        if len(tags) < 2:
            raise SystemExit(f"Need at least 2 run folders under {runs_dir}; found {len(tags)}")
        new_tag, baseline_tag = tags[0], tags[1]
        print(f"auto-last-two: new_tag={new_tag} baseline_tag={baseline_tag}", file=sys.stderr)
    else:
        if not args.new_tag or not args.baseline_tag:
            raise SystemExit("Provide --new-tag and --baseline-tag, or use --auto-last-two")
        new_tag, baseline_tag = args.new_tag, args.baseline_tag

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (runs_dir / new_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_root = runs_dir / new_tag
    base_root = runs_dir / baseline_tag
    if not new_root.is_dir():
        raise SystemExit(f"Missing new run directory: {new_root}")
    if not base_root.is_dir():
        raise SystemExit(f"Missing baseline run directory: {base_root}")

    def tag_for_variant(side: str, variant: str) -> str:
        """Resolve which folder name holds CSVs for this variant and side."""
        if side == "new":
            if variant == "best_prauc":
                return args.new_best_prauc_tag or new_tag
            if variant == "best_mcmae":
                return args.new_best_mcmae_tag or new_tag
        else:
            if variant == "best_prauc":
                return args.baseline_best_prauc_tag or baseline_tag
            if variant == "best_mcmae":
                return args.baseline_best_mcmae_tag or baseline_tag
        return new_tag if side == "new" else baseline_tag

    for variant in args.variants:
        nt = tag_for_variant("new", variant)
        bt = tag_for_variant("baseline", variant)
        new_v_root = runs_dir / nt
        base_v_root = runs_dir / bt
        if nt != new_tag or bt != baseline_tag:
            print(
                f"[{variant}] CSV roots: new={nt} (under {runs_dir.name}/), baseline={bt}",
                file=sys.stderr,
            )
        if not new_v_root.is_dir():
            print(f"[{variant}] Skip: missing new directory: {new_v_root}", file=sys.stderr)
            continue
        if not base_v_root.is_dir():
            print(f"[{variant}] Skip: missing baseline directory: {base_v_root}", file=sys.stderr)
            continue

        over_n = new_v_root / f"overall_metrics_by_split_{variant}.csv"
        over_b = base_v_root / f"overall_metrics_by_split_{variant}.csv"
        pl_n = new_v_root / f"per_label_metrics_by_split_{variant}.csv"
        pl_b = base_v_root / f"per_label_metrics_by_split_{variant}.csv"

        if not (over_n.is_file() and over_b.is_file()):
            print(
                f"[{variant}] Skip: missing overall CSV pair "
                f"({over_n.name} and/or {over_b.name})",
                file=sys.stderr,
            )
            continue
        if not (pl_n.is_file() and pl_b.is_file()):
            print(
                f"[{variant}] Skip: missing per-label CSV pair "
                f"({pl_n.name} and/or {pl_b.name})",
                file=sys.stderr,
            )
            continue

        df_on = _load_eval_csv(over_n)
        df_ob = _load_eval_csv(over_b)
        df_pn = _load_eval_csv(pl_n)
        df_pb = _load_eval_csv(pl_b)

        schema_ok = (
            _validate_schema(over_n, "new overall", df_on, ["split"], variant)
            and _validate_schema(over_b, "baseline overall", df_ob, ["split"], variant)
            and _validate_schema(pl_n, "new per_label", df_pn, ["split", "label"], variant)
            and _validate_schema(pl_b, "baseline per_label", df_pb, ["split", "label"], variant)
        )
        if not schema_ok:
            print(
                f"[{variant}] Skip: fix CSVs (re-run NB04 save cell) or check you opened the right files.",
                file=sys.stderr,
            )
            continue

        long_o, notes_o = _merge_long(
            df_on,
            df_ob,
            merge_keys=["split"],
            variant=variant,
            table="overall",
            skip_rel_change=frozenset(),
            new_tag_label=nt,
            baseline_tag_label=bt,
        )
        long_p, notes_p = _merge_long(
            df_pn,
            df_pb,
            merge_keys=["split", "label"],
            variant=variant,
            table="per_label",
            skip_rel_change=SKIP_REL_CHANGE_COLS,
            new_tag_label=nt,
            baseline_tag_label=bt,
        )

        for n in notes_o + notes_p:
            print(n, file=sys.stderr)

        _label_set_diagnostic(df_pn, df_pb, variant)

        out_over = out_dir / f"compare_to_{baseline_tag}_overall_{variant}.csv"
        out_pl = out_dir / f"compare_to_{baseline_tag}_per_label_{variant}.csv"
        long_o.to_csv(out_over, index=False)
        long_p.to_csv(out_pl, index=False)
        print(f"Wrote {out_over}")
        print(f"Wrote {out_pl}")

        _print_summary(long_o, variant)


if __name__ == "__main__":
    main()
