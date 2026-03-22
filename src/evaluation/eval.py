"""
Table Evaluation Script  (v2 — with full diagnostics)
======================================================
Evaluates VLM-predicted HTML tables against PubTabNet OTSL ground truth.

Inputs:
  - row_*.json  : PubTabNet ground truth (OTSL + cell tokens)
  - doc_*.md    : VLM prediction (markdown-wrapped HTML)

Metrics:
  - TEDS        : Tree Edit Distance Similarity (structure + content)
  - TEDS-Struct : Tree Edit Distance Similarity (structure only)
  - Cell P/R/F1 : Cell-text multiset precision / recall / F1
"""

import json
import re
from collections import Counter

from apted import APTED, Config
from bs4 import BeautifulSoup, NavigableString, Tag


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 64):
    print()
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


def _sub(title: str):
    print(f"\n  ▸ {title}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ground-truth reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_gt_html(row: dict, struct_only: bool = False) -> str:
    """
    Reconstruct the full HTML table from PubTabNet token format.

    'html'     — flat list of structural tokens
                 e.g. ['<thead>', '<tr>', '<td>', '</td>', '<td', ' colspan="2"', '>', ...]
    'cells[0]' — flat list of {tokens, bbox} dicts, one per </td>, in order

    We walk the token list and inject cell text immediately before each '</td>'.
    If struct_only=True we skip the injection (empty cells for TEDS-Struct).
    """
    html_tokens = row["html"]
    cells       = row["cells"][0]

    cell_idx   = 0
    html_parts = []

    for token in html_tokens:
        if token == "</td>":
            if (not struct_only) and cell_idx < len(cells):
                raw = "".join(cells[cell_idx]["tokens"])
                # Strip inline HTML formatting tags (<b>, <i>, etc.)
                text = re.sub(r"<[^>]+>", "", raw)
                html_parts.append(text)
            cell_idx += 1
            html_parts.append("</td>")
        else:
            html_parts.append(token)

    return f"<table>{''.join(html_parts)}</table>"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_html_from_markdown(md_text: str) -> str:
    """
    Pull the HTML table out of a markdown string.
    Tries fenced code block first, then bare <table>…</table>.
    """
    fence = re.search(r"```(?:html)?\s*(.*?)```", md_text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    bare = re.search(r"(<table[\s\S]*?</table>)", md_text, re.IGNORECASE | re.DOTALL)
    if bare:
        return bare.group(1).strip()

    raise ValueError("No HTML table found in markdown input.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTML normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_html(html_str: str, struct_only: bool = False) -> str:
    """
    Canonical form for comparison:
      • Keep only structural tags: table / thead / tbody / tfoot / tr / td / th
      • Keep only colspan and rowspan attributes; drop everything else
      • Remove colspan="1" / rowspan="1" (they are the default)
      • Collapse whitespace inside cells
      • If struct_only=True: wipe all cell text
    """
    soup  = BeautifulSoup(html_str, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> element found.")

    KEEP_TAGS  = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}
    KEEP_ATTRS = {"colspan", "rowspan"}

    def _clean(tag: Tag):
        for child in list(tag.children):
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            if child.name not in KEEP_TAGS:
                child.unwrap()
            else:
                _clean(child)

        if hasattr(tag, "attrs"):
            tag.attrs = {k: v for k, v in tag.attrs.items() if k in KEEP_ATTRS}
            for attr in ("colspan", "rowspan"):
                if tag.attrs.get(attr) in ("1", 1):
                    del tag.attrs[attr]

        if struct_only and tag.name in ("td", "th"):
            tag.clear()
            return

        if (not struct_only) and tag.name in ("td", "th"):
            text = tag.get_text(separator=" ", strip=True)
            tag.clear()
            if text:
                tag.string = text

    _clean(table)
    return str(table)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEDS
# ─────────────────────────────────────────────────────────────────────────────

class TableTree:
    def __init__(self, tag: str, text: str = "", attrs: dict = None):
        self.tag      = tag
        self.text     = text.strip() if text else ""
        self.attrs    = attrs or {}
        self.children: list["TableTree"] = []

    def __repr__(self):
        return f"TableTree({self.tag!r}, text={self.text!r}, n_children={len(self.children)})"


def _html_to_tree(html_str: str) -> TableTree | None:
    soup      = BeautifulSoup(html_str, "html.parser")
    table_tag = soup.find("table")
    if table_tag is None:
        return None

    def _convert(bs_node) -> TableTree | None:
        if isinstance(bs_node, str):
            text = bs_node.strip()
            return TableTree("__text__", text=text) if text else None
        node = TableTree(
            tag=bs_node.name,
            attrs={k: str(v) for k, v in bs_node.attrs.items()},
        )
        for child in bs_node.children:
            c = _convert(child)
            if c is not None:
                node.children.append(c)
        return node

    return _convert(table_tag)


class TEDSConfig(Config):
    def rename(self, n1: TableTree, n2: TableTree) -> float:
        if n1.tag != n2.tag:       return 1.0
        if n1.text != n2.text:     return 1.0
        if n1.attrs != n2.attrs:   return 1.0
        return 0.0

    def children(self, node: TableTree):
        return node.children


def _count_nodes(node: TableTree) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def compute_teds(
    pred_html:   str,
    gt_html:     str,
    struct_only: bool = False,
    verbose:     bool = False,
) -> float:
    """
    Compute TEDS (or TEDS-Struct when struct_only=True).

    Formula
    -------
        TEDS = 1 - EditDist(T_pred, T_gt) / max(|T_pred|, |T_gt|)

    The edit distance is computed with the APTED algorithm.
    Node costs: insert = delete = rename = 1 (rename = 0 when nodes are identical).
    """
    label = "TEDS-Struct" if struct_only else "TEDS"

    try:
        pred_norm = normalize_html(pred_html, struct_only=struct_only)
        gt_norm   = normalize_html(gt_html,   struct_only=struct_only)
    except Exception as e:
        print(f"    [WARN] normalisation failed: {e}")
        return 0.0

    pred_tree = _html_to_tree(pred_norm)
    gt_tree   = _html_to_tree(gt_norm)

    if pred_tree is None or gt_tree is None:
        print(f"    [WARN] tree conversion failed")
        return 0.0

    n_pred = _count_nodes(pred_tree)
    n_gt   = _count_nodes(gt_tree)
    denom  = max(n_pred, n_gt)

    apted = APTED(pred_tree, gt_tree, TEDSConfig())
    ted   = apted.compute_edit_distance()

    score = 1.0 - (ted / denom) if denom > 0 else 1.0

    if verbose:
        _sub(f"{label} calculation")
        print(f"      Tree nodes (pred)      : {n_pred}")
        print(f"      Tree nodes (GT)        : {n_gt}")
        print(f"      max(|T_pred|, |T_gt|)  : {denom}")
        print(f"      Edit distance (TED)    : {ted}")
        print(f"      Formula                : 1 - {ted} / {denom}")
        print(f"      {label:<22}= {score:.6f}")

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cell-level P / R / F1
# ─────────────────────────────────────────────────────────────────────────────

def extract_cells(html_str: str, include_empty: bool = False) -> list[str]:
    """
    Return a list of normalised cell-text strings from an HTML table.

    Parameters
    ----------
    include_empty : if False (default), cells whose text is empty after
                    stripping are excluded.  Setting include_empty=True
                    restores the old behaviour and is the root cause of
                    Precision = Recall = F1 when both tables have the
                    same number of empty cells.
    """
    soup  = BeautifulSoup(html_str, "html.parser")
    cells = []
    for tag in soup.find_all(["td", "th"]):
        text = tag.get_text(separator=" ", strip=True).lower()
        if text or include_empty:
            cells.append(text)
    return cells


def compute_cell_f1(
    pred_html: str,
    gt_html:   str,
    verbose:   bool = False,
) -> dict:
    """
    Multiset precision / recall / F1 over cell text strings.

    Why multiset?
    -------------
    A table can have repeated values (e.g. many cells containing "0" or "-").
    Using a plain set would treat all occurrences as one match; using a multiset
    (Counter) counts each occurrence separately, which is more faithful.

    Formula
    -------
        TP = |pred_cells ∩ gt_cells|   (multiset intersection — sum of mins)
        FP = |pred_cells − gt_cells|   (cells predicted but not in GT)
        FN = |gt_cells   − pred_cells| (GT cells absent from prediction)

        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 · Precision · Recall / (Precision + Recall)

    Note on empty cells
    -------------------
    Empty cells (<td></td>) produce the string "".  If both tables have the
    same number of empty cells they all land in TP and P = R = F1 exactly.
    We therefore EXCLUDE empty-string cells from the comparison by default.
    """
    pred_cells = Counter(extract_cells(pred_html, include_empty=False))
    gt_cells   = Counter(extract_cells(gt_html,   include_empty=False))

    tp = sum((pred_cells & gt_cells).values())   # multiset intersection
    fp = sum((pred_cells - gt_cells).values())   # extra predictions
    fn = sum((gt_cells - pred_cells).values())   # missed GT cells

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    if verbose:
        _sub("Cell-level F1 calculation")
        print(f"      Non-empty cells (pred)   : {sum(pred_cells.values())}")
        print(f"      Non-empty cells (GT)     : {sum(gt_cells.values())}")
        print()
        print(f"      TP (matched cells)       : {tp}")
        print(f"      FP (extra in pred)       : {fp}")
        print(f"      FN (missing from pred)   : {fn}")
        print()
        print(f"      Precision = {tp} / ({tp}+{fp}) = {precision:.6f}")
        print(f"      Recall    = {tp} / ({tp}+{fn}) = {recall:.6f}")
        if (precision + recall) > 0:
            print(f"      F1        = 2·{precision:.4f}·{recall:.4f} / ({precision:.4f}+{recall:.4f}) = {f1:.6f}")
        else:
            print(f"      F1        = 0.0 (precision + recall = 0)")

        # Show top FP and FN to help diagnose errors
        fp_cells = list((pred_cells - gt_cells).elements())
        fn_cells = list((gt_cells - pred_cells).elements())
        if fp_cells:
            sample = fp_cells[:8]
            print(f"\n      Top FP cells (predicted but wrong):")
            for c in sample:
                print(f"        '{c}'")
            if len(fp_cells) > 8:
                print(f"        ... and {len(fp_cells)-8} more")
        if fn_cells:
            sample = fn_cells[:8]
            print(f"\n      Top FN cells (GT cells missed):")
            for c in sample:
                print(f"        '{c}'")
            if len(fn_cells) > 8:
                print(f"        ... and {len(fn_cells)-8} more")

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    gt_path:  str,
    pred_path: str,
    verbose:  bool = True,
) -> dict:
    """
    Run the full evaluation pipeline for one (GT, prediction) pair.

    Parameters
    ----------
    gt_path   : path to a PubTabNet row_*.json file
    pred_path : path to a VLM output markdown file (doc_*.md)
    verbose   : print step-by-step calculation details
    """

    # ── Load ──────────────────────────────────────────────────────
    with open(gt_path, encoding="utf-8") as f:
        gt_row = json.load(f)["row"]

    with open(pred_path, encoding="utf-8") as f:
        md_text = f.read()

    # ── Reconstruct GT ────────────────────────────────────────────
    _banner("Step 1 — Ground-Truth Reconstruction")
    gt_html_full   = reconstruct_gt_html(gt_row, struct_only=False)
    gt_html_struct = reconstruct_gt_html(gt_row, struct_only=True)

    # Count raw cells to show context
    n_gt_cells_all   = sum(1 for t in gt_row["html"] if t == "</td>")
    n_gt_cells_empty = sum(
        1 for c in gt_row["cells"][0]
        if not re.sub(r"<[^>]+>", "", "".join(c["tokens"])).strip()
    )

    print(f"  File          : {gt_row['filename']}")
    print(f"  Table size    : {gt_row['rows']} rows × {gt_row['cols']} cols")
    print(f"  Total cells   : {n_gt_cells_all}")
    print(f"  Empty cells   : {n_gt_cells_empty}  "
          f"← these are EXCLUDED from Cell-F1 to prevent P = R = F1 artefact")
    print(f"  GT HTML chars : {len(gt_html_full)}")

    # ── Extract prediction ────────────────────────────────────────
    _banner("Step 2 — Prediction Extraction")
    pred_html = extract_html_from_markdown(md_text)

    soup_pred = BeautifulSoup(pred_html, "html.parser")
    n_pred_cells_all   = len(soup_pred.find_all(["td", "th"]))
    n_pred_cells_empty = sum(
        1 for tag in soup_pred.find_all(["td", "th"])
        if not tag.get_text(strip=True)
    )
    print(f"  Source        : {pred_path}")
    print(f"  HTML chars    : {len(pred_html)}")
    print(f"  Total cells   : {n_pred_cells_all}")
    print(f"  Empty cells   : {n_pred_cells_empty}")

    # ── Compute metrics ───────────────────────────────────────────
    _banner("Step 3 — TEDS")
    teds = compute_teds(pred_html, gt_html_full, struct_only=False, verbose=verbose)

    _banner("Step 4 — TEDS-Struct")
    teds_struct = compute_teds(pred_html, gt_html_struct, struct_only=True, verbose=verbose)

    _banner("Step 5 — Cell-Level F1")
    cell_scores = compute_cell_f1(pred_html, gt_html_full, verbose=verbose)

    # ── Summary ───────────────────────────────────────────────────
    _banner("Summary")
    print(f"  {'Metric':<30} {'Score':>10}")
    print(f"  {'─'*30} {'─'*10}")
    print(f"  {'TEDS (struct + content)':<30} {teds:>10.4f}")
    print(f"  {'TEDS-Struct (struct only)':<30} {teds_struct:>10.4f}")
    print(f"  {'Cell Precision':<30} {cell_scores['precision']:>10.4f}")
    print(f"  {'Cell Recall':<30} {cell_scores['recall']:>10.4f}")
    print(f"  {'Cell F1':<30} {cell_scores['f1']:>10.4f}")
    print()
    print(f"  TP={cell_scores['tp']}  FP={cell_scores['fp']}  FN={cell_scores['fn']}")

    return {
        "gt_filename": gt_row["filename"],
        "gt_rows":     gt_row["rows"],
        "gt_cols":     gt_row["cols"],
        "TEDS":        teds,
        "TEDS-Struct": teds_struct,
        "Cell-P":      cell_scores["precision"],
        "Cell-R":      cell_scores["recall"],
        "Cell-F1":     cell_scores["f1"],
        "TP":          cell_scores["tp"],
        "FP":          cell_scores["fp"],
        "FN":          cell_scores["fn"],
    }


def main():
    GT_PATH   = "PubTabNet_OTSL_train_20/row_16.json"
    PRED_PATH = "experiments/output/otsl_3/doc_0.md"

    result = evaluate(GT_PATH, PRED_PATH, verbose=True)

    out_path = "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()