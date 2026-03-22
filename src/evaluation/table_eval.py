"""
table_eval.py  v2
=================
Evaluation framework for VLM table extraction.

Handles:
  - Simple tables (all fcel, no merges)
  - Complex tables with merged cells (lcel / ucel / ecel / xcel in OTSL)
  - VLM predictions as raw HTML strings (with inline styles, markdown fences)

Comparison strategy for merged-cell tables
------------------------------------------
When GT contains merged cells the pred HTML (which lacks rowspan/colspan) has
a different row/col structure.  We compare:
  1. OTSL structure   — token accuracy on the OTSL sequence (always meaningful)
  2. HTML structure   — sequence similarity on structural tokens
  3. fcel content     — only GT cells with OTSL token 'fcel' carry unique content;
                        we extract those in reading order and match them against
                        pred cells in reading order.
  4. Grid dimensions  — reported but not penalised when merges are present.

Usage
-----
  python table_eval.py --gt row_4.json --pred doc_0.html
  python table_eval.py --demo
  python table_eval.py --gt data.jsonl --pred preds.json --out results.json
"""

from __future__ import annotations

import re, json, warnings
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from collections import Counter

try:
    from rich.console import Console
    from rich.table import Table as RichTable
    from rich import box
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False; console = None

try:
    import zss; _ZSS = True
except ImportError:
    _ZSS = False

_FMT_PAT = re.compile(r"</?(?:b|i|u|strong|em|s|sub|sup)>")

# OTSL tokens that consume a cell dict from the flat cells list
_OTSL_CELL_TOKENS  = {"fcel", "ecel"}
# OTSL tokens that carry unique content (not a reference to another cell)
_OTSL_CONTENT_TOKS = {"fcel"}


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CellResult:
    row: int
    col: int
    gt_text: str
    pred_text: str
    gt_otsl_tok: str        # fcel / ecel / lcel / ucel / xcel / unknown
    exact_match: bool
    char_similarity: float
    had_formatting: bool


@dataclass
class EvalConfig:
    strip_formatting: bool = True
    normalise_html: bool = True
    case_sensitive: bool = True
    compute_ted: bool = True
    html_tags_to_strip: tuple = (
        "<thead>", "</thead>", "<tbody>", "</tbody>",
        "<tfoot>", "</tfoot>",
    )
    formatting_pattern: str = r"</?(?:b|i|u|strong|em|s|sub|sup)>"


@dataclass
class TableSample:
    gt_cells: list          # 2-D list[list[str]] — plain text per GT grid slot
    gt_otsl_grid: list      # 2-D list[list[str]] — OTSL token per GT grid slot
    pred_cells_raw: list    # flat list[dict] with tokens key  OR  list[list[str]]
    pred_is_html: bool
    gt_otsl: list
    pred_otsl: list
    gt_html: list
    pred_html: list
    gt_rows: int
    gt_cols: int
    pred_rows: int
    pred_cols: int
    has_merged_cells: bool  # True if OTSL contains lcel/ucel/xcel
    sample_id: str = "sample"


@dataclass
class EvalReport:
    sample_id: str
    pred_source: str = "unknown"
    has_merged_cells: bool = False

    # Dimensions
    dim_match: bool = False
    gt_rows: int = 0; gt_cols: int = 0
    pred_rows: int = 0; pred_cols: int = 0

    # OTSL
    otsl_exact: bool = False
    otsl_sequence_sim: float = 0.0
    otsl_token_accuracy: float = 0.0
    otsl_token_distribution: dict = field(default_factory=dict)

    # HTML structure
    html_exact: bool = False
    html_sequence_sim: float = 0.0
    html_gt_len: int = 0
    html_pred_len: int = 0
    html_ted: Optional[float] = None
    html_normalised_note: str = ""

    # Cell content
    cell_results: list = field(default_factory=list)
    cell_exact_accuracy: float = 0.0
    cell_mean_char_sim: float = 0.0
    total_cells_compared: int = 0
    cells_skipped_merged: int = 0   # lcel/ucel/xcel slots skipped
    cells_with_formatting: int = 0

    composite_score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HTML prediction parser
# ═══════════════════════════════════════════════════════════════════════════════

class _TableHTMLParser(HTMLParser):
    """Parse a VLM HTML table string into flat rows of cell text."""
    def __init__(self):
        super().__init__()
        self._rows = []; self._cur_row = []; self._cell_buf = []
        self._in_cell = False; self._depth = 0
        self.structure_tokens = []

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in ("thead","tbody","tfoot"):
            self.structure_tokens.append(f"<{t}>")
        elif t == "tr":
            self._cur_row = []; self.structure_tokens.append("<tr>")
        elif t in ("td","th"):
            self._in_cell = True; self._depth = 0; self._cell_buf = []
            self.structure_tokens.append("<td>")
        elif self._in_cell:
            self._depth += 1

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("thead","tbody","tfoot"):
            self.structure_tokens.append(f"</{t}>")
        elif t == "tr":
            self._rows.append(self._cur_row); self.structure_tokens.append("</tr>")
        elif t in ("td","th"):
            if self._in_cell:
                self._cur_row.append("".join(self._cell_buf).strip())
                self._in_cell = False; self._depth = 0; self._cell_buf = []
            self.structure_tokens.append("</td>")
        elif self._in_cell and self._depth > 0:
            self._depth -= 1

    def handle_data(self, data):
        if self._in_cell: self._cell_buf.append(data)

    @property
    def rows(self): return self._rows
    @property
    def n_rows(self): return len(self._rows)
    @property
    def n_cols(self): return max((len(r) for r in self._rows), default=0)


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_pred_html_string(html_str: str):
    """Parse VLM HTML prediction. Returns (rows_2d, struct_tokens, n_rows, n_cols)."""
    p = _TableHTMLParser()
    try:
        p.feed(_strip_fence(html_str))
    except Exception as exc:
        warnings.warn(f"HTML parse error: {exc}")
        return [], [], 0, 0
    if not p.rows:
        warnings.warn("HTML parser found no rows — check prediction format.")
    return p.rows, p.structure_tokens, p.n_rows, p.n_cols


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class _Utils:
    @staticmethod
    def seq_sim(a, b): return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def tok_acc(pred, gt):
        if not gt and not pred: return 1.0
        if not gt or not pred:  return 0.0
        return sum(p==g for p,g in zip(pred,gt)) / max(len(pred),len(gt))

    @staticmethod
    def norm_html(tokens, cfg):
        return [t for t in tokens if t not in cfg.html_tags_to_strip] if cfg.normalise_html else tokens

    @staticmethod
    def clean(text, cfg):
        s = re.sub(cfg.formatting_pattern, "", text).strip() if cfg.strip_formatting else text.strip()
        return s.lower() if not cfg.case_sensitive else s

    @staticmethod
    def had_fmt(text, cfg):
        return bool(re.search(cfg.formatting_pattern, text))

    @staticmethod
    def norm_grid(grid, cfg):
        processed, flags = [], []
        for row in grid:
            p_row = []
            for cell in row:
                flags.append(_Utils.had_fmt(cell, cfg))
                p_row.append(_Utils.clean(cell, cfg))
            processed.append(p_row)
        return processed, flags

    @staticmethod
    def flatten_token_cells(raw_cells, cols, cfg):
        texts, flags = [], []
        for cell in raw_cells:
            raw  = "".join(cell.get("tokens", []))
            had  = bool(re.search(cfg.formatting_pattern, raw))
            text = _Utils.clean(raw, cfg)
            texts.append(text); flags.append(had)
        grid = [texts[i:i+cols] for i in range(0, len(texts), cols)]
        return grid, flags

    @staticmethod
    def html_to_tree(tokens):
        if not _ZSS: return None
        try:
            stack = [zss.Node("root")]
            for tok in tokens:
                if tok.startswith("</"): 
                    if len(stack) > 1: stack.pop()
                elif tok.startswith("<"):
                    n = zss.Node(tok.strip("<>").split()[0]); stack[-1].addkid(n); stack.append(n)
            return stack[0]
        except Exception as e:
            warnings.warn(f"TED tree error: {e}"); return None

    @staticmethod
    def ted(gt_toks, pred_toks):
        if not _ZSS: return None
        gt_t = _Utils.html_to_tree(gt_toks); pr_t = _Utils.html_to_tree(pred_toks)
        if gt_t is None or pr_t is None: return None
        try: return float(zss.simple_distance(gt_t, pr_t))
        except Exception as e: warnings.warn(f"TED error: {e}"); return None


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loader — handles flat-OTSL schema (merged cells)
# ═══════════════════════════════════════════════════════════════════════════════

def _reconstruct_gt_grid(cells_flat: list, otsl: list, cols: int):
    """
    Build a 2-D GT grid from a flat cell list and the OTSL sequence.

    fcel / ecel tokens consume the next cell dict.
    lcel / ucel / xcel tokens are spanned slots — they get text "[merged]".
    """
    cell_idx = 0
    flat_text  = []
    flat_otsl  = []
    for tok in otsl:
        if tok == "nl": continue
        if tok in _OTSL_CELL_TOKENS:
            raw  = "".join(cells_flat[cell_idx].get("tokens", [])) if cell_idx < len(cells_flat) else ""
            text = _FMT_PAT.sub("", raw).strip()
            cell_idx += 1
        else:
            text = "[merged]"
        flat_text.append(text)
        flat_otsl.append(tok)

    gt_cells  = [flat_text[i:i+cols] for i in range(0, len(flat_text), cols)]
    otsl_grid = [flat_otsl[i:i+cols] for i in range(0, len(flat_otsl), cols)]
    return gt_cells, otsl_grid


def _parse_row_dict(row, row_idx):
    raw_cells = row["cells"]
    cols      = int(row["cols"])
    rows      = int(row["rows"])
    filename  = row.get("filename", f"row_{row_idx}")
    sample_id = f"{filename}  [idx={row_idx}]"
    otsl      = row.get("otsl", [])

    # Detect schema: flat-OTSL (one outer wrapper row) vs simple 2D grid
    is_flat_schema = (
        isinstance(raw_cells, list)
        and len(raw_cells) == 1
        and isinstance(raw_cells[0], list)
        and raw_cells[0]
        and isinstance(raw_cells[0][0], dict)
    )

    if is_flat_schema:
        cells_flat    = raw_cells[0]
        gt_cells, otsl_grid = _reconstruct_gt_grid(cells_flat, otsl, cols)
        pred_cells_raw = cells_flat
    else:
        # Simple 2D grid (previous simple tables)
        gt_cells = [
            [_FMT_PAT.sub("", "".join(c.get("tokens", []))).strip() for c in r]
            for r in raw_cells
        ]
        otsl_grid = None
        pred_cells_raw = [c for r in raw_cells for c in r]

    # Build otsl_grid as flat if None (simple tables have no merges)
    if otsl_grid is None:
        flat_toks = [t for t in otsl if t != "nl"]
        otsl_grid = [flat_toks[i:i+cols] for i in range(0, len(flat_toks), cols)]

    has_merged = any(t in {"lcel","ucel","xcel"} for t in otsl)

    return TableSample(
        gt_cells       = gt_cells,
        gt_otsl_grid   = otsl_grid,
        pred_cells_raw = pred_cells_raw,
        pred_is_html   = False,
        gt_otsl        = otsl,
        pred_otsl      = otsl,
        gt_html        = row.get("html", []),
        pred_html      = row.get("html_restored", []),
        gt_rows        = rows, gt_cols = cols,
        pred_rows      = rows, pred_cols = cols,
        has_merged_cells = has_merged,
        sample_id      = sample_id,
    )


def _parse_raw_json(data, source_path):
    samples = []
    if isinstance(data, dict) and "row" in data:
        samples.append(_parse_row_dict(data["row"], data.get("row_idx", 0)))
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "row" in entry:
                samples.append(_parse_row_dict(entry["row"], entry.get("row_idx", 0)))
            elif isinstance(entry, dict) and "cells" in entry:
                samples.append(_parse_row_dict(entry, 0))
    elif isinstance(data, dict) and "cells" in data:
        samples.append(_parse_row_dict(data, 0))
    else:
        raise ValueError(f"Unrecognised JSON structure in '{source_path}'.")
    return samples


def load_samples(path: str):
    """Load GT samples from a .json or .jsonl PubTabNet/OTSL file."""
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"GT file not found: '{path}'")
    samples = []
    if p.suffix.lower() == ".jsonl":
        with open(p, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line: continue
                try: samples.extend(_parse_raw_json(json.loads(line), f"{path}:{lineno}"))
                except json.JSONDecodeError as e: warnings.warn(f"Bad JSON line {lineno}: {e}")
    else:
        with open(p, encoding="utf-8") as fh:
            samples.extend(_parse_raw_json(json.load(fh), path))
    if not samples: raise ValueError(f"No valid samples in '{path}'.")
    print(f"Loaded {len(samples)} GT sample(s) from '{path}'.")
    return samples


def load_predictions(path: str) -> dict:
    """
    Load VLM predictions → dict { filename → html_string }.

    Formats:
      .txt/.md/.html  — single prediction (key = stem)
      .json           — { "file.png": "<table>…</table>" }
      .jsonl          — { "filename": "…", "prediction": "…" }
    """
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"Prediction file not found: '{path}'")
    if p.suffix.lower() in (".txt", ".md", ".html"):
        pred_map = {p.stem: p.read_text(encoding="utf-8")}
        print(f"Loaded 1 prediction from '{path}'  (key='{p.stem}').")
        return pred_map
    if p.suffix.lower() == ".jsonl":
        pred_map = {}
        with open(p, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    k = obj.get("filename") or obj.get("sample_id") or obj.get("id")
                    v = obj.get("prediction") or obj.get("html") or obj.get("output")
                    if k and v: pred_map[k] = v
                    else: warnings.warn(f"Line {ln}: missing filename/prediction keys.")
                except json.JSONDecodeError as e: warnings.warn(f"Bad JSON line {ln}: {e}")
        print(f"Loaded {len(pred_map)} prediction(s) from '{path}'.")
        return pred_map
    with open(p, encoding="utf-8") as fh: data = json.load(fh)
    if not isinstance(data, dict): raise ValueError(f"Expected JSON object in '{path}'.")
    print(f"Loaded {len(data)} prediction(s) from '{path}'.")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TableEvaluator:
    """
    Evaluate TableSamples. Handles both simple and merged-cell tables.

    Usage
    -----
    samples   = load_samples("gt.json")
    preds     = load_predictions("pred.html")
    evaluator = TableEvaluator()
    reports   = evaluator.evaluate_batch(samples, preds)
    evaluator.print_aggregate(reports)
    """
    def __init__(self, config=None):
        self.config = config or EvalConfig()

    def evaluate(self, sample: TableSample, pred_map=None) -> EvalReport:
        report = EvalReport(sample_id=sample.sample_id,
                            has_merged_cells=sample.has_merged_cells)

        if pred_map:
            html_str = (
                pred_map.get(sample.sample_id)
                or pred_map.get(Path(sample.sample_id.split("[")[0].strip()).name)
                or pred_map.get(Path(sample.sample_id.split("[")[0].strip()).stem)
            )
            if html_str:
                cells_2d, struct_toks, n_rows, n_cols = parse_pred_html_string(html_str)
                sample = replace(sample,
                    pred_cells_raw=cells_2d, pred_is_html=True,
                    pred_html=struct_toks, pred_rows=n_rows, pred_cols=n_cols)
                report.pred_source = "html_string"
            else:
                warnings.warn(f"[{sample.sample_id}] Not in pred_map → html_restored fallback.")
                report.pred_source = "html_restored (fallback)"
        else:
            report.pred_source = "html_string" if sample.pred_is_html else "html_restored"

        self._eval_dimensions(sample, report)
        self._eval_otsl(sample, report)
        self._eval_html(sample, report)
        self._eval_cells(sample, report)
        self._compute_composite(report)
        return report

    def evaluate_batch(self, samples, pred_map=None):
        return [self.evaluate(s, pred_map) for s in samples]

    def print_report(self, report):
        if _RICH: self._rich_report(report)
        else:     self._plain_report(report)

    def print_aggregate(self, reports):
        if not reports: print("No reports."); return
        if _RICH: self._rich_aggregate(reports)
        else:     self._plain_aggregate(reports)

    def to_dict(self, report):
        d = {k: v for k, v in report.__dict__.items() if k != "cell_results"}
        d["cell_results"] = [cr.__dict__ for cr in report.cell_results]
        return d

    def save_json(self, reports, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([self.to_dict(r) for r in reports], fh, indent=2, ensure_ascii=False)
        print(f"Saved {len(reports)} report(s) to '{path}'")

    # ── Evaluation steps ───────────────────────────────────────────────────────

    def _eval_dimensions(self, s, r):
        r.gt_rows=s.gt_rows; r.gt_cols=s.gt_cols
        r.pred_rows=s.pred_rows; r.pred_cols=s.pred_cols
        r.dim_match = (s.gt_rows==s.pred_rows) and (s.gt_cols==s.pred_cols)

    def _eval_otsl(self, s, r):
        r.otsl_exact          = s.gt_otsl == s.pred_otsl
        r.otsl_sequence_sim   = _Utils.seq_sim(s.gt_otsl, s.pred_otsl)
        r.otsl_token_accuracy = _Utils.tok_acc(s.pred_otsl, s.gt_otsl)
        non_nl = [t for t in s.gt_otsl if t != "nl"]
        r.otsl_token_distribution = dict(Counter(non_nl))

    def _eval_html(self, s, r):
        cfg = self.config
        gt_n   = _Utils.norm_html(s.gt_html, cfg)
        pred_n = _Utils.norm_html(s.pred_html, cfg)
        r.html_exact        = gt_n == pred_n
        r.html_sequence_sim = _Utils.seq_sim(gt_n, pred_n)
        r.html_gt_len       = len(s.gt_html)
        r.html_pred_len     = len(s.pred_html)
        only_gt   = set(s.gt_html)   - set(pred_n)
        only_pred = set(s.pred_html) - set(gt_n)
        if only_gt or only_pred:
            r.html_normalised_note = (
                f"Only in GT: {only_gt or '—'}  |  Only in pred: {only_pred or '—'}"
            )
        if cfg.compute_ted:
            r.html_ted = _Utils.ted(gt_n, pred_n)

    def _eval_cells(self, s, r):
        cfg = self.config

        # ── Resolve pred grid ──────────────────────────────────────────────────
        if s.pred_is_html:
            pred_grid, fmt_flags = _Utils.norm_grid(s.pred_cells_raw, cfg)
        else:
            pred_grid, fmt_flags = _Utils.flatten_token_cells(
                s.pred_cells_raw, s.pred_cols, cfg)

        # ── For merged-cell tables: compare only fcel positions ────────────────
        # Extract fcel GT texts and pred texts in reading order.
        # Pred rows are flat (no rowspan) so we scan them in order too.
        if s.has_merged_cells:
            self._eval_cells_merged(s, pred_grid, fmt_flags, r)
        else:
            self._eval_cells_simple(s, pred_grid, fmt_flags, r)

    def _eval_cells_simple(self, s, pred_grid, fmt_flags, r):
        """Standard cell-by-cell comparison for simple (non-merged) tables."""
        results, exact_count, sim_sum, total = [], 0, 0.0, 0
        n_rows = min(len(s.gt_cells), len(pred_grid))
        for ri in range(n_rows):
            gt_row, pred_row = s.gt_cells[ri], pred_grid[ri]
            n_cols   = min(len(gt_row), len(pred_row))
            flat_idx = ri * max(len(pred_row), 1)
            for ci in range(n_cols):
                gt_t   = _Utils.clean(gt_row[ci], self.config)
                pred_t = pred_row[ci]
                sim    = SequenceMatcher(None, pred_t, gt_t).ratio()
                had_f  = fmt_flags[flat_idx+ci] if (flat_idx+ci)<len(fmt_flags) else False
                otsl_t = s.gt_otsl_grid[ri][ci] if s.gt_otsl_grid and ri<len(s.gt_otsl_grid) else "fcel"
                results.append(CellResult(
                    row=ri, col=ci,
                    gt_text=gt_row[ci], pred_text=pred_row[ci],
                    gt_otsl_tok=otsl_t,
                    exact_match=(gt_t==pred_t),
                    char_similarity=round(sim,4),
                    had_formatting=had_f,
                ))
                exact_count += int(gt_t==pred_t); sim_sum += sim; total += 1
        self._set_cell_stats(r, results, total, exact_count, sim_sum, fmt_flags)

    def _eval_cells_merged(self, s, pred_grid, fmt_flags, r):
        """
        For merged-cell tables compare in reading order:
          GT  : fcel positions only (unique-content cells), skipping lcel/ecel/ucel/xcel
          Pred: all cells in reading order (flattened pred rows)

        This is the fairest comparison because:
          - lcel/ucel slots are just duplicates of their span source
          - ecel slots are intentionally empty
          - The pred has no span info, so its cell count ≠ GT grid cell count
        """
        # Collect GT fcel texts in reading order
        gt_fcel: list[tuple[int,int,str]] = []   # (row, col, text)
        skipped = 0
        for ri, row in enumerate(s.gt_cells):
            otsl_row = s.gt_otsl_grid[ri] if s.gt_otsl_grid and ri < len(s.gt_otsl_grid) else []
            for ci, gt_text in enumerate(row):
                otsl_t = otsl_row[ci] if ci < len(otsl_row) else "fcel"
                if otsl_t in _OTSL_CONTENT_TOKS:
                    gt_fcel.append((ri, ci, gt_text))
                else:
                    skipped += 1

        # Flatten pred grid to a 1-D list of cell texts
        pred_flat = [cell for row in pred_grid for cell in row]

        results, exact_count, sim_sum, total = [], 0, 0.0, 0
        n_compare = min(len(gt_fcel), len(pred_flat))

        for k in range(n_compare):
            ri, ci, gt_raw = gt_fcel[k]
            pred_text = pred_flat[k]
            gt_t  = _Utils.clean(gt_raw, self.config)
            sim   = SequenceMatcher(None, pred_text, gt_t).ratio()
            had_f = fmt_flags[k] if k < len(fmt_flags) else False
            otsl_t = s.gt_otsl_grid[ri][ci] if s.gt_otsl_grid else "fcel"
            results.append(CellResult(
                row=ri, col=ci,
                gt_text=gt_raw, pred_text=pred_text,
                gt_otsl_tok=otsl_t,
                exact_match=(gt_t==pred_text),
                char_similarity=round(sim,4),
                had_formatting=had_f,
            ))
            exact_count += int(gt_t==pred_text); sim_sum += sim; total += 1

        r.cells_skipped_merged = skipped
        self._set_cell_stats(r, results, total, exact_count, sim_sum, fmt_flags)

    @staticmethod
    def _set_cell_stats(r, results, total, exact_count, sim_sum, fmt_flags):
        r.cell_results          = results
        r.total_cells_compared  = total
        r.cell_exact_accuracy   = exact_count / total if total else 0.0
        r.cell_mean_char_sim    = sim_sum / total if total else 0.0
        r.cells_with_formatting = sum(fmt_flags)

    def _compute_composite(self, r):
        r.composite_score = round(
            0.50 * r.cell_exact_accuracy
            + 0.25 * r.otsl_token_accuracy
            + 0.15 * r.html_sequence_sim
            + 0.10 * float(r.dim_match), 4)

    # ── Plain printing ─────────────────────────────────────────────────────────

    def _plain_report(self, r):
        sep = "=" * 76
        print(f"\n{sep}")
        print(f"  TABLE EVAL — {r.sample_id}")
        print(f"  Pred source: {r.pred_source}  |  Merged cells: {'yes' if r.has_merged_cells else 'no'}")
        print(sep)

        print(f"\n[1] Grid Dimensions")
        print(f"    GT   : {r.gt_rows}r × {r.gt_cols}c")
        print(f"    Pred : {r.pred_rows}r × {r.pred_cols}c  {'✓' if r.dim_match else '✗ (expected if merged)'}")

        print(f"\n[2] OTSL Structure")
        print(f"    Exact: {'✓' if r.otsl_exact else '✗'}  Token acc: {r.otsl_token_accuracy:.4f}  Seq sim: {r.otsl_sequence_sim:.4f}")
        print(f"    Token distribution: {r.otsl_token_distribution}")

        print(f"\n[3] HTML Structure")
        print(f"    Exact: {'✓' if r.html_exact else '✗'}  Seq sim: {r.html_sequence_sim:.4f}  Len GT/Pred: {r.html_gt_len}/{r.html_pred_len}")
        if r.html_ted is not None:
            print(f"    TED  : {r.html_ted:.1f}")
        elif not _ZSS:
            print(f"    TED  : N/A (pip install zss)")
        if r.html_normalised_note:
            print(f"    Note : {r.html_normalised_note}")

        note = ""
        if r.has_merged_cells:
            note = f"  (fcel only — {r.cells_skipped_merged} merged/empty slots skipped)"
        print(f"\n[4] Cell Content{note}")
        print(f"    Exact accuracy : {r.cell_exact_accuracy:.4f}  ({r.cell_exact_accuracy*100:.1f}%)")
        print(f"    Mean char sim  : {r.cell_mean_char_sim:.4f}")
        print(f"    Total compared : {r.total_cells_compared}")
        print(f"    Cells w/ fmt   : {r.cells_with_formatting}")
        print(f"\n    {'Row':<5}{'Col':<5}{'OTSL':<7}{'GT':<32}{'Predicted':<32}{'✓/✗':<5}{'Sim'}")
        print("    " + "─" * 86)
        for cr in r.cell_results:
            mark = "✓" if cr.exact_match else "✗"
            fmt  = "[fmt]" if cr.had_formatting else ""
            print(f"    {cr.row:<5}{cr.col:<5}{cr.gt_otsl_tok:<7}"
                  f"{cr.gt_text:<32}{(cr.pred_text+fmt):<32}{mark:<5}{cr.char_similarity}")

        print(f"\n[5] Composite Score : {r.composite_score:.4f}  ({r.composite_score*100:.1f}%)")
        print(sep + "\n")

    def _plain_aggregate(self, reports):
        n   = len(reports)
        sep = "=" * 76
        print(f"\n{sep}")
        print(f"  AGGREGATE  ({n} sample{'s' if n>1 else ''})")
        print(sep)
        rows = [
            ("Dimension match (%)",        sum(r.dim_match for r in reports)/n*100),
            ("OTSL exact (%)",             sum(r.otsl_exact for r in reports)/n*100),
            ("OTSL token accuracy",        sum(r.otsl_token_accuracy for r in reports)/n),
            ("HTML normalised exact (%)",  sum(r.html_exact for r in reports)/n*100),
            ("HTML sequence sim",          sum(r.html_sequence_sim for r in reports)/n),
            ("Cell exact accuracy",        sum(r.cell_exact_accuracy for r in reports)/n),
            ("Cell mean char sim",         sum(r.cell_mean_char_sim for r in reports)/n),
            ("Composite score",            sum(r.composite_score for r in reports)/n),
        ]
        ted_vals = [r.html_ted for r in reports if r.html_ted is not None]
        if ted_vals: rows.append(("Mean TED", sum(ted_vals)/len(ted_vals)))
        for k, v in rows: print(f"    {k:<40} {v:.4f}")
        print(sep + "\n")

    # ── Rich printing ──────────────────────────────────────────────────────────

    def _rich_report(self, r):
        c = console
        c.rule(f"[bold cyan]{r.sample_id}[/]  [dim]{r.pred_source}  merged={'yes' if r.has_merged_cells else 'no'}[/]")
        dim_col = "green" if r.dim_match else "yellow"
        c.print(f"\n[bold]Dims[/] GT {r.gt_rows}×{r.gt_cols} → Pred {r.pred_rows}×{r.pred_cols} [{dim_col}]{'✓' if r.dim_match else '≠'}[/]")
        c.print(f"[bold]OTSL[/] exact=[{'green' if r.otsl_exact else 'red'}]{'✓' if r.otsl_exact else '✗'}[/] tok_acc={r.otsl_token_accuracy:.3f} seq_sim={r.otsl_sequence_sim:.3f}  dist={r.otsl_token_distribution}")
        ted_s  = f"  TED={r.html_ted:.1f}" if r.html_ted is not None else ""
        html_c = "green" if r.html_exact else "yellow"
        c.print(f"[bold]HTML[/] exact=[{html_c}]{'✓' if r.html_exact else '✗'}[/] seq_sim={r.html_sequence_sim:.3f} len={r.html_gt_len}/{r.html_pred_len}{ted_s}")
        if r.html_normalised_note: c.print(f"  [dim]{r.html_normalised_note}[/]")

        note_title = "Cell Content"
        if r.has_merged_cells:
            note_title += f" (fcel only — {r.cells_skipped_merged} merged/empty skipped)"
        tbl = RichTable(title=note_title, box=box.SIMPLE, show_header=True, header_style="bold")
        for col_name in ("Row","Col","OTSL","GT","Predicted","Exact","Sim","Fmt"):
            tbl.add_column(col_name)
        for cr in r.cell_results:
            tbl.add_row(
                str(cr.row), str(cr.col), cr.gt_otsl_tok,
                cr.gt_text, cr.pred_text,
                "[green]✓[/]" if cr.exact_match else "[red]✗[/]",
                f"{cr.char_similarity:.3f}",
                "[yellow]fmt[/]" if cr.had_formatting else "",
            )
        c.print(tbl)
        c.print(f"[bold]Cell exact[/] {r.cell_exact_accuracy*100:.1f}%  "
                f"[bold]Char sim[/] {r.cell_mean_char_sim:.3f}  "
                f"[bold]Composite[/] [cyan]{r.composite_score*100:.1f}%[/]\n")

    def _rich_aggregate(self, reports):
        n   = len(reports)
        tbl = RichTable(title=f"Aggregate ({n} samples)", box=box.ROUNDED)
        tbl.add_column("Metric", style="bold"); tbl.add_column("Value", justify="right")
        rows = [
            ("Dim match",       f"{sum(r.dim_match for r in reports)/n*100:.1f}%"),
            ("OTSL exact",      f"{sum(r.otsl_exact for r in reports)/n*100:.1f}%"),
            ("OTSL token acc",  f"{sum(r.otsl_token_accuracy for r in reports)/n:.4f}"),
            ("HTML exact",      f"{sum(r.html_exact for r in reports)/n*100:.1f}%"),
            ("HTML seq sim",    f"{sum(r.html_sequence_sim for r in reports)/n:.4f}"),
            ("Cell exact",      f"{sum(r.cell_exact_accuracy for r in reports)/n:.4f}"),
            ("Cell char sim",   f"{sum(r.cell_mean_char_sim for r in reports)/n:.4f}"),
            ("Composite",       f"{sum(r.composite_score for r in reports)/n:.4f}"),
        ]
        ted_vals = [r.html_ted for r in reports if r.html_ted is not None]
        if ted_vals: rows.append(("Mean TED", f"{sum(ted_vals)/len(ted_vals):.2f}"))
        for label, val in rows: tbl.add_row(label, val)
        console.print(tbl)


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in demo (simple table, no merges)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_demo():
    demo_row = {
        "filename": "PMC4517499_004_00.png", "cols": 7, "rows": 4,
        "cells": [[
            {"tokens": ["<b>","T","y","p","e"," ","o","f"," ","d","e","l","a","y","</b>"], "bbox":[1,4,46,13,2]},
            {"tokens": ["<b>","M","e","a","n","</b>"],   "bbox":[89,4,109,13,2]},
            {"tokens": ["<b>","S","D","</b>"],           "bbox":[116,4,127,13,2]},
            {"tokens": ["<b>","M","e","d","i","a","n","</b>"], "bbox":[138,4,163,13,2]},
            {"tokens": ["<b>","I","Q","R","</b>"],       "bbox":[170,4,183,13,2]},
            {"tokens": ["<b>","M","i","n","</b>"],       "bbox":[200,4,214,13,2]},
            {"tokens": ["<b>","M","a","x","</b>"],       "bbox":[221,4,236,13,2]},
            {"tokens": ["P","a","t","i","e","n","t"," ","d","e","l","a","y"], "bbox":[1,17,44,27,2]},
            {"tokens": ["5","5",".","3"], "bbox":[94,17,109,27,2]},
            {"tokens": ["4","0",".","0"], "bbox":[116,17,131,27,2]},
            {"tokens": ["5","9"],         "bbox":[138,17,147,27,2]},
            {"tokens": ["5","-","1","2","3"], "bbox":[170,17,190,27,2]},
            {"tokens": ["5"],             "bbox":[200,17,206,27,2]},
            {"tokens": ["1","9","8"],     "bbox":[221,17,235,27,2]},
            {"tokens": ["H","e","a","l","t","h","c","a","r","e"," ","s","e","r","v","i","c","e","s"," ","d","e","l","a","y"], "bbox":[1,31,83,41,2]},
            {"tokens": ["7","6",".","5"], "bbox":[94,31,109,41,2]},
            {"tokens": ["9","1",".","2"], "bbox":[116,31,131,41,2]},
            {"tokens": ["4","5"],         "bbox":[138,31,147,41,2]},
            {"tokens": ["3","8","-","1","2","8"], "bbox":[170,31,194,41,2]},
            {"tokens": ["0"],             "bbox":[200,31,206,41,2]},
            {"tokens": ["3","7","1"],     "bbox":[221,31,235,41,2]},
            {"tokens": ["T","o","t","a","l"," ","d","i","a","g","n","o","s","t","i","c"," ","d","e","l","a","y"], "bbox":[1,45,73,55,2]},
            {"tokens": ["1","3","1",".","4"], "bbox":[90,45,109,55,2]},
            {"tokens": ["9","4",".","3"],     "bbox":[116,45,131,55,2]},
            {"tokens": ["1","0","4"],         "bbox":[138,45,151,55,2]},
            {"tokens": ["1","7","-","1","8","7"], "bbox":[170,45,194,55,2]},
            {"tokens": ["1","4"],             "bbox":[200,45,210,55,2]},
            {"tokens": ["4","0","1"],         "bbox":[221,45,235,55,2]},
        ]],
        "otsl": ["fcel"]*7+["nl"]+["fcel"]*7+["nl"]+["fcel"]*7+["nl"]+["fcel"]*7+["nl"],
        "html": ["<thead>","<tr>"]+["<td>","</td>"]*7+["</tr>","</thead>","<tbody>"]+
                (["<tr>"]+["<td>","</td>"]*7+["</tr>"])*3+["</tbody>"],
        "html_restored": [],
    }
    pred_html = """```markdown
<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Type of delay</td><td style='text-align: center; word-wrap: break-word;'>Mean</td><td style='text-align: center; word-wrap: break-word;'>SD</td><td style='text-align: center; word-wrap: break-word;'>Median</td><td style='text-align: center; word-wrap: break-word;'>IQR</td><td style='text-align: center; word-wrap: break-word;'>Min</td><td style='text-align: center; word-wrap: break-word;'>Max</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Patient delay</td><td style='text-align: center; word-wrap: break-word;'>55.3</td><td style='text-align: center; word-wrap: break-word;'>40.0</td><td style='text-align: center; word-wrap: break-word;'>59</td><td style='text-align: center; word-wrap: break-word;'>5-123</td><td style='text-align: center; word-wrap: break-word;'>5</td><td style='text-align: center; word-wrap: break-word;'>198</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Healthcare services delay</td><td style='text-align: center; word-wrap: break-word;'>76.5</td><td style='text-align: center; word-wrap: break-word;'>91.2</td><td style='text-align: center; word-wrap: break-word;'>45</td><td style='text-align: center; word-wrap: break-word;'>38-128</td><td style='text-align: center; word-wrap: break-word;'>0</td><td style='text-align: center; word-wrap: break-word;'>371</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Total diagnostic delay</td><td style='text-align: center; word-wrap: break-word;'>131.4</td><td style='text-align: center; word-wrap: break-word;'>94.3</td><td style='text-align: center; word-wrap: break-word;'>104</td><td style='text-align: center; word-wrap: break-word;'>17-187</td><td style='text-align: center; word-wrap: break-word;'>14</td><td style='text-align: center; word-wrap: break-word;'>401</td></tr></table>
```"""
    sample   = _parse_row_dict(demo_row, 0)
    pred_map = {"PMC4517499_004_00.png": pred_html}
    return sample, pred_map


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        prog="table_eval.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate VLM table extraction vs PubTabNet/OTSL GT.",
        epilog="""
Prediction formats (--pred):
  .txt/.md/.html  — single HTML prediction for one table
  .json           — { "filename.png": "<table>…</table>" }
  .jsonl          — { "filename": "…", "prediction": "…" }

Examples:
  python table_eval.py --demo
  python table_eval.py --gt row_4.json --pred doc_0.html
  python table_eval.py --gt train.jsonl --pred preds.json --out results.json
        """,
    )
    parser.add_argument("--gt",   metavar="PATH")
    parser.add_argument("--pred", metavar="PATH")
    parser.add_argument("--out",  metavar="PATH")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-strip-fmt",      action="store_true")
    parser.add_argument("--no-normalise-html", action="store_true")
    parser.add_argument("--case-insensitive",  action="store_true")
    parser.add_argument("--no-ted",            action="store_true")
    parser.add_argument("--verbose",           action="store_true")
    args = parser.parse_args()

    if not args.demo and not args.gt:
        parser.print_help(); sys.exit(0)

    config = EvalConfig(
        strip_formatting = not args.no_strip_fmt,
        normalise_html   = not args.no_normalise_html,
        case_sensitive   = not args.case_insensitive,
        compute_ted      = not args.no_ted,
    )
    evaluator = TableEvaluator(config)

    if args.demo:
        print("Running built-in demo …\n")
        sample, pred_map = _build_demo()
        reports = evaluator.evaluate_batch([sample], pred_map)
        evaluator.print_report(reports[0])
        evaluator.save_json(reports, args.out or "eval_results_demo.json")
        sys.exit(0)

    samples  = load_samples(args.gt)
    pred_map = load_predictions(args.pred) if args.pred else None
    reports  = evaluator.evaluate_batch(samples, pred_map)

    if args.verbose or len(reports) == 1:
        for r in reports: evaluator.print_report(r)
    if len(reports) > 1:
        evaluator.print_aggregate(reports)

    out_path = args.out or str(Path(args.gt).with_suffix("")) + "_eval.json"
    evaluator.save_json(reports, out_path)