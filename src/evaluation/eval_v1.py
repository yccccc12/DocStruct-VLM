"""
Table Evaluation Script
=======================
Evaluates VLM-predicted HTML tables against PubTabNet OTSL ground truth.

Inputs:
  - row_4.json         : PubTabNet ground truth (OTSL + cell tokens)
  - doc_0.md           : VLM prediction (markdown-wrapped HTML)  [source A]
  - pruned_result_0.json: VLM prediction (JSON with block_content) [source B]

Metrics:
  - TEDS       : Tree Edit Distance Similarity (structure + content)
  - TEDS-Struct: Tree Edit Distance Similarity (structure only)
"""

import json
import re
import sys
from collections import deque
from copy import deepcopy

from apted import APTED, Config
from apted.helpers import Tree
from bs4 import BeautifulSoup
from lxml import etree


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ground-truth reconstruction from PubTabNet token format
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_gt_html(row: dict) -> str:
    """
    Reconstruct the full HTML table (with cell text) from PubTabNet format.

    The 'html' field is a flat list of structural tokens such as:
        ['<thead>', '<tr>', '<td>', '</td>', '<td', ' colspan="2"', '>', '</td>', ...]
    Cell content is stored in 'cells[0]' as a list of {tokens, bbox} objects,
    one entry per </td>, in document order.

    Strategy: walk the token list and inject cell content immediately
    before every '</td>' token.
    """
    html_tokens = row["html"]
    cells       = row["cells"][0]          # flat list, same length as </td> count

    cell_idx  = 0
    html_parts = []

    for token in html_tokens:
        if token == "</td>":
            # Inject cell content
            if cell_idx < len(cells):
                cell_text = "".join(cells[cell_idx]["tokens"])
                # Strip inline formatting tags so we compare plain text
                cell_text = re.sub(r"<[^>]+>", "", cell_text)
                html_parts.append(cell_text)
            cell_idx += 1
            html_parts.append("</td>")
        else:
            html_parts.append(token)

    inner_html = "".join(html_parts)
    return f"<table>{inner_html}</table>"


def reconstruct_gt_html_struct_only(row: dict) -> str:
    """
    Same as above but leaves all cells empty (for TEDS-Struct).
    """
    html_tokens = row["html"]
    return f"<table>{''.join(html_tokens)}</table>"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract predicted HTML from VLM outputs
# ─────────────────────────────────────────────────────────────────────────────

def extract_html_from_markdown(md_text: str) -> str:
    """
    Extract the raw HTML table from a markdown string.
    Handles both:
      - ```html ... ``` fenced blocks
      - Bare <table> ... </table> (no fence)
    """
    # Try fenced code block first
    fence_match = re.search(
        r"```(?:html)?\s*(.*?)```",
        md_text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Fall back: grab everything between first <table and last </table>
    table_match = re.search(
        r"(<table[\s\S]*?</table>)",
        md_text,
        re.IGNORECASE | re.DOTALL,
    )
    if table_match:
        return table_match.group(1).strip()

    raise ValueError("No HTML table found in markdown input.")


# def extract_html_from_pruned_json(data: dict) -> str:
#     """
#     Extract the HTML table from the PaddleX / docling 'pruned_result' JSON.
#     Looks for the first entry in parsing_res_list whose block_label == 'table'.
#     """
#     for block in data.get("parsing_res_list", []):
#         if block.get("block_label") == "table":
#             html = block["block_content"]
#             # Ensure it is wrapped in <table> tags
#             if not html.strip().lower().startswith("<table"):
#                 html = f"<table>{html}</table>"
#             return html
#     raise ValueError("No 'table' block found in pruned_result JSON.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTML normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_html(html_str: str, struct_only: bool = False) -> str:
    """
    Parse HTML with BeautifulSoup, then re-serialise to a canonical form.

    Canonicalization steps:
      - Keep only <table>, <thead>, <tbody>, <tfoot>, <tr>, <td>, <th> tags
      - Preserve colspan / rowspan attributes (drop all others)
      - Normalise whitespace inside cells
      - If struct_only=True, strip all cell text content
    """
    from bs4 import NavigableString, Tag

    soup = BeautifulSoup(html_str, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> element found in HTML string.")

    KEEP_TAGS  = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}
    KEEP_ATTRS = {"colspan", "rowspan"}

    def _clean(tag: Tag):
        # Iterate over a snapshot of children
        for child in list(tag.children):
            if isinstance(child, NavigableString):
                continue                       # leave text nodes alone for now
            if not isinstance(child, Tag):
                continue
            if child.name not in KEEP_TAGS:
                child.unwrap()                 # promote grandchildren, remove tag
            else:
                _clean(child)

        # Strip disallowed attributes
        if hasattr(tag, "attrs"):
            tag.attrs = {
                k: v
                for k, v in tag.attrs.items()
                if k in KEEP_ATTRS
            }
            # Normalise span="1" → omit (it is the default)
            for attr in ("colspan", "rowspan"):
                if tag.attrs.get(attr) in ("1", 1):
                    del tag.attrs[attr]

        # Optionally wipe cell text
        if struct_only and tag.name in ("td", "th"):
            tag.clear()
            return

        # Normalise text whitespace inside cells
        if (not struct_only) and tag.name in ("td", "th"):
            text = tag.get_text(separator=" ", strip=True)
            tag.clear()
            if text:
                tag.string = text

    _clean(table)
    return str(table)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEDS implementation
# ─────────────────────────────────────────────────────────────────────────────

class TableTree:
    """
    Lightweight node for APTED tree edit distance.
    Each node stores:
      - tag   : HTML tag name (or '__text__' for text nodes)
      - text  : cell text content (empty for structural nodes)
      - attrs : frozenset of (attr, val) pairs relevant to structure
    """

    def __init__(self, tag: str, text: str = "", attrs: dict = None):
        self.tag      = tag
        self.text     = text.strip() if text else ""
        self.attrs    = attrs or {}
        self.children: list["TableTree"] = []

    def bracket(self) -> str:
        """APTED bracket notation."""
        label = self.tag
        if self.text:
            label += f":{self.text}"
        for k in sorted(self.attrs):
            label += f"[{k}={self.attrs[k]}]"
        kids = "".join(c.bracket() for c in self.children)
        return f"{{{label}{kids}}}"

    def __repr__(self):
        return f"TableTree({self.tag!r}, text={self.text!r}, children={len(self.children)})"


def _html_to_tree(html_str: str) -> TableTree:
    """Convert a normalised HTML string into a TableTree."""
    soup = BeautifulSoup(html_str, "html.parser")
    table_tag = soup.find("table")
    if table_tag is None:
        raise ValueError("No <table> found.")

    def _convert(bs_node) -> TableTree | None:
        # Text node
        if isinstance(bs_node, str):
            text = bs_node.strip()
            if text:
                node = TableTree("__text__", text=text)
                return node
            return None

        # Element node
        node = TableTree(
            tag=bs_node.name,
            attrs={k: str(v) for k, v in bs_node.attrs.items()},
        )
        for child in bs_node.children:
            child_node = _convert(child)
            if child_node is not None:
                node.children.append(child_node)
        return node

    return _convert(table_tag)


class TEDSConfig(Config):
    """
    APTED configuration for TableTree nodes.

    rename cost  = 1  if nodes differ, 0 if same
    insert/delete cost = 1
    """

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        if node1.tag != node2.tag:
            return 1.0
        if node1.text != node2.text:
            return 1.0
        if node1.attrs != node2.attrs:
            return 1.0
        return 0.0

    def children(self, node: TableTree):
        return node.children


def _count_nodes(node: TableTree) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def compute_teds(pred_html: str, gt_html: str, struct_only: bool = False) -> float:
    """
    Compute TEDS between predicted and ground-truth HTML.

    TEDS = 1 - (edit_distance / max(|T_pred|, |T_gt|))

    Args:
        pred_html   : prediction HTML string (will be normalised internally)
        gt_html     : ground-truth HTML string (will be normalised internally)
        struct_only : if True, ignore cell text (compute TEDS-Struct)

    Returns:
        float in [0, 1], higher is better.
    """
    try:
        pred_norm = normalize_html(pred_html, struct_only=struct_only)
        gt_norm   = normalize_html(gt_html,   struct_only=struct_only)
    except Exception as e:
        print(f"  [WARN] HTML normalisation failed: {e}")
        return 0.0

    try:
        pred_tree = _html_to_tree(pred_norm)
        gt_tree   = _html_to_tree(gt_norm)
    except Exception as e:
        print(f"  [WARN] Tree conversion failed: {e}")
        return 0.0

    if pred_tree is None or gt_tree is None:
        return 0.0

    n_pred = _count_nodes(pred_tree)
    n_gt   = _count_nodes(gt_tree)

    apted   = APTED(pred_tree, gt_tree, TEDSConfig())
    ted     = apted.compute_edit_distance()

    denom = max(n_pred, n_gt)
    if denom == 0:
        return 1.0

    return 1.0 - (ted / denom)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cell-level precision / recall / F1 (bonus metric)
# ─────────────────────────────────────────────────────────────────────────────

def extract_cells(html_str: str) -> list[str]:
    """Return a list of normalised cell text strings from an HTML table."""
    soup = BeautifulSoup(html_str, "html.parser")
    cells = []
    for tag in soup.find_all(["td", "th"]):
        text = tag.get_text(separator=" ", strip=True).lower()
        cells.append(text)
    return cells


def compute_cell_f1(pred_html: str, gt_html: str):
    """
    Compute precision, recall and F1 at the cell-text level.
    Treats the two bags of cell strings as multisets.
    """
    from collections import Counter

    pred_cells = Counter(extract_cells(pred_html))
    gt_cells   = Counter(extract_cells(gt_html))

    tp = sum((pred_cells & gt_cells).values())
    fp = sum((pred_cells - gt_cells).values())
    fn = sum((gt_cells - pred_cells).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    GT_PATH      = "PubTabNet_OTSL_train_20/row_16.json"
    MD_PATH      = "experiments/output/otsl_3/doc_0.md"
    # JSON_PATH    = "experiments/output/otsl/pruned_result_0.json"

    # ── Load files ────────────────────────────────────────────────
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)["row"]

    with open(MD_PATH, encoding="utf-8") as f:
        md_text = f.read()

    # with open(JSON_PATH, encoding="utf-8") as f:
    #     json_data = json.load(f)

    # ── Reconstruct ground truth HTML ─────────────────────────────
    print_section("Ground Truth Reconstruction")
    gt_html_full   = reconstruct_gt_html(gt_data)
    gt_html_struct = reconstruct_gt_html_struct_only(gt_data)
    print(f"  GT filename  : {gt_data['filename']}")
    print(f"  GT table size: {gt_data['rows']} rows × {gt_data['cols']} cols")
    print(f"  GT HTML chars: {len(gt_html_full)}")

    # ── Extract predictions ───────────────────────────────────────
    print_section("Prediction Extraction")

    # Source A: doc_0.md
    pred_html_md = extract_html_from_markdown(md_text)
    print(f"  [doc_0.md]            HTML chars: {len(pred_html_md)}")

    # Source B: pruned_result_0.json
    # pred_html_json = extract_html_from_pruned_json(json_data)
    # print(f"  [pruned_result_0.json] HTML chars: {len(pred_html_json)}")

    # ── Evaluate both sources ─────────────────────────────────────
    sources = {
        "doc_0.md (VLM markdown output)": pred_html_md,
        # "pruned_result_0.json (PaddleX)":  pred_html_json,
    }

    results = {}
    for name, pred_html in sources.items():
        print_section(f"Evaluation: {name}")

        teds        = compute_teds(pred_html, gt_html_full,   struct_only=False)
        teds_struct = compute_teds(pred_html, gt_html_struct, struct_only=True)
        cell_scores = compute_cell_f1(pred_html, gt_html_full)

        results[name] = {
            "TEDS":         teds,
            "TEDS-Struct":  teds_struct,
            "Cell P":       cell_scores["precision"],
            "Cell R":       cell_scores["recall"],
            "Cell F1":      cell_scores["f1"],
        }

        print(f"  TEDS         (structure + content)  : {teds:.4f}")
        print(f"  TEDS-Struct  (structure only)       : {teds_struct:.4f}")
        print(f"  Cell Precision                      : {cell_scores['precision']:.4f}")
        print(f"  Cell Recall                         : {cell_scores['recall']:.4f}")
        print(f"  Cell F1                             : {cell_scores['f1']:.4f}")

    # ── Summary table ─────────────────────────────────────────────
    print_section("Summary")
    header = f"{'Source':<40} {'TEDS':>8} {'TEDS-S':>8} {'Cell-P':>8} {'Cell-R':>8} {'Cell-F1':>8}"
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        short = name[:38]
        print(
            f"{short:<40} "
            f"{scores['TEDS']:>8.4f} "
            f"{scores['TEDS-Struct']:>8.4f} "
            f"{scores['Cell P']:>8.4f} "
            f"{scores['Cell R']:>8.4f} "
            f"{scores['Cell F1']:>8.4f}"
        )

    # ── Save results ──────────────────────────────────────────────
    output = {
        "ground_truth": {
            "filename": gt_data["filename"],
            "rows": gt_data["rows"],
            "cols": gt_data["cols"],
        },
        "results": results,
    }
    out_path = "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {out_path}")

    # ── Also save the reconstructed GT HTML for inspection ────────
    gt_out = "gt_reconstructed.html"
    with open(gt_out, "w", encoding="utf-8") as f:
        f.write(f"<html><body>{gt_html_full}</body></html>")
    print(f"  GT HTML saved to: {gt_out}")


if __name__ == "__main__":
    main()