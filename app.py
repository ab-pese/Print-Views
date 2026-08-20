"""
Sheet Region Exporter
======================
Reads a drawing sheet PDF (Revit-exported or similar), auto-detects the
schedules (tables), views (drafting-view drawings w/ title + scale callout),
and general notes block on each page, lists them in reading order with a
checkbox next to each, and lets you export only the checked items into a
brand new PDF at a paper size / orientation you choose (Letter or Tabloid,
Portrait or Landscape).

Detection is heuristic (built for Revit sheet conventions: boxed schedule
tables with a merged title row, and view captions of the form
    TITLE TEXT
    3/8" = 1'-0"
directly above the drawing). It will not be perfect on every sheet layout,
so every detected box is editable (and new ones addable) in the table below
the preview -- nothing is locked in until you click Generate.

Run with:  streamlit run sheet_region_exporter.py
"""

import concurrent.futures
import hashlib
import io
import json
import os
import re
import tempfile
import traceback

import numpy as np
import pandas as pd
import pdfplumber
import pymupdf  # PyMuPDF
import streamlit as st
from PIL import Image, ImageDraw
from scipy import ndimage

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

if HAS_CANVAS:
    # streamlit-drawable-canvas (unmaintained since ~2022) calls the internal
    # helper streamlit.elements.image.image_to_url(image, width, clamp,
    # channels, output_format, image_id). Modern Streamlit moved that helper
    # to streamlit.elements.lib.image_utils and changed its signature to take
    # a LayoutConfig instead of a plain width, which raises an AttributeError
    # on import. Patch a signature-compatible shim back onto the old location
    # so the component still works; if some future Streamlit version breaks
    # this shim too, just disable the canvas feature instead of crashing.
    try:
        import streamlit.elements.image as _st_image_mod
        if not hasattr(_st_image_mod, "image_to_url"):
            from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
            from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

            def _image_to_url_shim(image, width, clamp, channels, output_format, image_id):
                return _new_image_to_url(
                    image, _LayoutConfig(width=width), clamp, channels, output_format, image_id
                )

            _st_image_mod.image_to_url = _image_to_url_shim
    except Exception:
        HAS_CANVAS = False

# ============================================================
# Local persistence (remembers checkbox selections + footer fields
# per-PDF across app restarts, keyed by a hash of the file contents)
# ============================================================
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sre_cache.json")


def _pdf_hash(pdf_bytes):
    return hashlib.sha256(pdf_bytes).hexdigest()[:20]


def _region_key(page, kind, title):
    return f"{page}|{kind}|{title}"


def _load_cache():
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


# ============================================================
# Session state init (stability pattern)
# ============================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.regions_df = pd.DataFrame(
        columns=["Include", "Page", "Kind", "Title", "X0", "Y0", "X1", "Y1"]
    )
    st.session_state.detected = False
    st.session_state.pdf_bytes = None
    st.session_state.pdf_name = None
    st.session_state.pdf_hash = None
    st.session_state.uploader_key = 0
    st.session_state.drawing_name = ""
    st.session_state.drawing_no = ""
    st.session_state.project_name = ""
    st.session_state.canvas_selection_key = None
    st.session_state.canvas_initial = None
    st.session_state.canvas_region_order = []

st.set_page_config(page_title="Sheet Region Exporter", layout="wide")

# ============================================================
# Constants / tunables
# ============================================================
PAPER_SIZES_PT = {
    "Letter (8.5 x 11)": (612, 792),
    "Tabloid (11 x 17)": (792, 1224),
}
KIND_COLORS = {
    "Schedule": (46, 96, 166),   # blue
    "View": (95, 158, 64),       # green
    "Notes": (196, 128, 30),     # orange
}


# ============================================================
# Detection (delegates to sre_detection.py, a standalone module with no
# Streamlit dependency, imported below)
# ============================================================
import sre_detection


def detect_all_pages(pdf_bytes, page_numbers, on_progress=None,
                      page_cache=None, force_rescan=False, max_workers=None):
    """page_numbers is a 1-indexed list of pages to scan.

    Scans pages in parallel using separate *processes* (not threads).
    pdfplumber's parsing is almost entirely pure Python, so it never
    releases the GIL -- a thread pool can't run it concurrently and only
    adds scheduling overhead. Separate processes each get their own
    interpreter/GIL, so this is what actually cuts wall-clock time on a
    70+ sheet batch (roughly in proportion to CPU cores available).

    The per-page work itself lives in sre_detection.detect_one_page_worker,
    in its own module with no Streamlit imports -- see that module's
    docstring for why (in short: Windows spawns worker processes by
    re-importing whatever module the target function lives in, and that
    must not be this Streamlit script itself).

    page_cache, if given, is a {page_number_str: [region dict, ...]} map of
    already-detected pages from a previous run of this same file (see the
    on-disk cache) -- those pages are reused instead of re-scanned unless
    force_rescan is True. Returns (all_regions, errors, updated_page_cache).
    """
    page_cache = dict(page_cache or {})
    all_regions = []
    errors = []
    to_scan = []
    for pno in page_numbers:
        key = str(pno)
        if not force_rescan and key in page_cache:
            all_regions.extend(page_cache[key])
        else:
            to_scan.append(pno)

    total = len(page_numbers)
    done = len(page_numbers) - len(to_scan)
    if on_progress:
        on_progress(done, total)

    def _record(pno, regions, err):
        nonlocal done
        if err:
            errors.append((pno, err))
        else:
            page_cache[str(pno)] = regions
            all_regions.extend(regions)
        done += 1
        if on_progress:
            on_progress(done, total)

    if to_scan:
        max_workers = max_workers or min((os.cpu_count() or 4), len(to_scan))
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(sre_detection.detect_one_page_worker, pdf_bytes, pno): pno
                    for pno in to_scan
                }
                for future in concurrent.futures.as_completed(futures):
                    pno, regions, err = future.result()
                    _record(pno, regions, err)
        except Exception:
            # Some hosts (locked-down sandboxes, certain restricted deploy
            # environments) don't allow spawning subprocesses at all -- fall
            # back to plain sequential scanning in-process rather than
            # failing the whole detection run.
            for pno in to_scan:
                pno2, regions, err = sre_detection.detect_one_page_worker(pdf_bytes, pno)
                _record(pno2, regions, err)

    return all_regions, errors, page_cache


# ============================================================
# Title-block field extraction (Drawing Name / Drawing No / Project Name)
# ============================================================
TB_LABEL_KEYWORDS = ["DRAWING NAME", "DRAWING NO", "PROJECT NO", "SHEET NO",
                     "DESIGN BY", "DRAWN BY", "CHECKED BY", "SCALE", "DATE",
                     "REVISIONS", "OF"]
_ADDRESS_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}\b")


def _is_tb_label(text):
    tu = text.strip().upper().rstrip(":.").strip()
    return any(tu == kw or tu.startswith(kw) for kw in TB_LABEL_KEYWORDS)


def _tb_expand(b, p):
    return (b[0] - p, b[1] - p, b[2] + p, b[3] + p)


def _tb_overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _find_value_near_label(spans, label_kw, pad=25):
    label_span = None
    for s in spans:
        tu = s["text"].strip().upper().rstrip(":.").strip()
        if tu == label_kw:
            label_span = s
            break
    if not label_span:
        return None
    cands = [s for s in spans if s is not label_span and not _is_tb_label(s["text"])
             and _tb_overlaps(_tb_expand(label_span["bbox"], pad), s["bbox"])]
    if not cands:
        return None
    return max(cands, key=lambda s: s["size"])["text"].strip()


def _extract_title_block_fields_from_page(page):
    """Core extraction logic, operating on an already-open PyMuPDF page --
    every sheet has its own Drawing Name / Drawing No, so this gets called
    once per exported sheet (via an already-open document) rather than
    re-opening the whole PDF per sheet."""
    result = {"drawing_name": "", "drawing_no": "", "project_name": ""}
    try:
        page_w, page_h = page.rect.width, page.rect.height
        d = page.get_text("dict")
        spans = []
        for block in d["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        spans.append({"text": span["text"], "bbox": span["bbox"], "size": span["size"]})

        dn = _find_value_near_label(spans, "DRAWING NAME")
        dno = _find_value_near_label(spans, "DRAWING NO")
        if dn:
            result["drawing_name"] = dn
        if dno:
            result["drawing_no"] = dno

        # Project name: title-block text usually lives in a narrow strip
        # along one edge of the sheet; look wherever a "DRAWING NO"/"DRAWING
        # NAME" label was actually found (fall back to the right-hand 15%
        # of the page, the common Revit convention, if no label matched).
        anchor = None
        for s in spans:
            tu = s["text"].strip().upper().rstrip(":.").strip()
            if tu in ("DRAWING NAME", "DRAWING NO"):
                anchor = s
                break
        if anchor:
            strip_x0 = anchor["bbox"][0] - 250
        else:
            strip_x0 = page_w * 0.85

        tb_spans = [s for s in spans if s["bbox"][0] > strip_x0]
        top_cluster = [s for s in tb_spans
                       if s["bbox"][1] < page_h * 0.28
                       and s["size"] >= 15  # project-name text is set noticeably larger than field labels/values
                       and not _ADDRESS_RE.search(s["text"])
                       and not _is_tb_label(s["text"])]
        if top_cluster:
            leftmost = min(top_cluster, key=lambda s: s["bbox"][0])
            result["project_name"] = leftmost["text"].strip()
    except Exception:
        pass
    return result


def extract_title_block_fields(pdf_bytes, page_index=0):
    """Best-effort auto-detect of Drawing Name / Drawing No / Project Name
    from a Revit-style title block (works even when that block's text is
    rotated 90 degrees along a vertical sheet edge). Any field it can't
    confidently find is returned as an empty string -- the UI lets you
    correct these before export."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        result = _extract_title_block_fields_from_page(doc[page_index])
        doc.close()
        return result
    except Exception:
        return {"drawing_name": "", "drawing_no": "", "project_name": ""}


# ============================================================
# Preview rendering
# ============================================================
def render_preview_image(pdf_bytes, page_index, regions_for_page, dpi=100):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)
    for r in regions_for_page:
        x0, y0, x1, y1 = r["bbox"]
        color = KIND_COLORS.get(r["kind"], (120, 120, 120))
        box = (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)
        draw.rectangle(box, outline=color, width=3)
        label = f"{r['kind']}: {r['title']}"
        text_y = max(box[1] - 16, 0)
        draw.rectangle((box[0], text_y, box[0] + 8 * len(label), text_y + 14), fill=color)
        draw.text((box[0] + 2, text_y), label, fill=(255, 255, 255))
    doc.close()
    return img


# ============================================================
# Output PDF composition (vector-quality crop via show_pdf_page)
# ============================================================
def _shelf_pack(regions, content_w, content_h, spacing, scale):
    """Greedy shelf (row) packing at a fixed uniform scale. Returns
    (rows, total_height) where rows is a list of lists of
    (region, scaled_w, scaled_h) in placement order."""
    rows, cur_row, cur_row_w, cur_row_h = [], [], 0.0, 0.0
    for r in regions:
        rw, rh = r["X1"] - r["X0"], r["Y1"] - r["Y0"]
        sw, sh = rw * scale, rh * scale
        add_w = sw + (spacing if cur_row else 0)
        if cur_row and (cur_row_w + add_w > content_w):
            rows.append((cur_row_h, cur_row))
            cur_row, cur_row_w, cur_row_h = [], 0.0, 0.0
            add_w = sw
        cur_row.append((r, sw, sh))
        cur_row_w += add_w
        cur_row_h = max(cur_row_h, sh)
    if cur_row:
        rows.append((cur_row_h, cur_row))
    total_h = sum(rh for rh, _ in rows) + spacing * max(len(rows) - 1, 0)
    return rows, total_h


def _best_single_page_scale(regions, content_w, content_h, spacing):
    """Binary search the largest uniform scale at which every selected
    region still fits on one page via shelf packing."""
    lo, hi = 0.0005, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2
        _, total_h = _shelf_pack(regions, content_w, content_h, spacing, mid)
        if total_h <= content_h:
            lo = mid
        else:
            hi = mid
    return lo


FOOTER_FONTSIZE = 8


def _draw_footer(page, w, h, margin, footer_fields):
    """Draws 'DRAWING NAME: x   DRAWING NO: x   PROJECT: x' along the
    bottom margin of a page, skipping any field left blank."""
    parts = []
    if footer_fields.get("project_name"):
        parts.append(f"PROJECT: {footer_fields['project_name']}")
    if footer_fields.get("drawing_name"):
        parts.append(f"DRAWING NAME: {footer_fields['drawing_name']}")
    if footer_fields.get("drawing_no"):
        parts.append(f"DRAWING NO: {footer_fields['drawing_no']}")
    if not parts:
        return
    text = "     |     ".join(parts)
    y = h - margin * 0.45
    page.draw_line((margin, y - FOOTER_FONTSIZE - 4), (w - margin, y - FOOTER_FONTSIZE - 4),
                    color=(0.75, 0.75, 0.75), width=0.5)
    page.insert_text((margin, y), text, fontsize=FOOTER_FONTSIZE, fontname="helv", color=(0.2, 0.2, 0.2))


def _group_by_sheet(regions):
    """Groups regions by their source sheet (input Page), preserving the
    order sheets first appear in. Each group becomes its own project --
    one sheet in, one output page out -- so items from different sheets
    never end up sharing an output page."""
    sheets = {}
    order = []
    for r in regions:
        key = int(r["Page"])
        if key not in sheets:
            sheets[key] = []
            order.append(key)
        sheets[key].append(r)
    return [sheets[key] for key in order]


def _dedupe_rows_by_name(rows):
    """One representative row per distinct (Kind, Title) -- first occurrence
    wins. Used to build a manual-layout *template* (one box per named
    view/schedule) instead of one box per individual sheet, so the same
    hand-arranged layout can be repeated across every sheet that has that
    name rather than cramming every sheet's copy onto one page."""
    seen = set()
    out = []
    for r in rows:
        key = (r["Kind"], r["Title"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_output_pdf(pdf_bytes, selected_regions, paper_key, orientation,
                      margin=36, spacing=18, footer_fields=None,
                      manual_dest=None):
    """Each source sheet (input PDF page) is treated as its own project:
    every sheet gets its own output page -- items from two different sheets
    are never packed onto the same output page together. Every sheet keeps
    its own Drawing Name / Drawing No in the footer (read fresh from that
    sheet's own title block); only Project Name is shared across all pages,
    from footer_fields. All the resulting pages land in one combined PDF.

    manual_dest, if given, is {source_sheet_page: [(region_dict, dest_dict), ...]}
    -- the exact per-view/schedule-name positions/sizes the user dragged and
    resized in the layout editor, as a single template that's repeated as
    its own output page for every sheet that has a region for that name
    (selected_regions is ignored in this mode -- each sheet's own region
    comes straight out of manual_dest instead). dest_dict is
    {'x','y','w','h'} in PDF points on one output page."""
    w, h = PAPER_SIZES_PT[paper_key]
    if orientation == "Landscape":
        w, h = h, w

    footer_fields = footer_fields or {}
    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()
    content_w = w - 2 * margin
    content_h = h - 2 * margin

    # Track page *numbers* (and each one's source sheet), not the Page
    # objects themselves: PyMuPDF orphans earlier Page wrappers (parent
    # becomes None) as soon as another page is added to the document, so
    # holding onto e.g. page 1's object while pages 2-72 get created and
    # then drawing on it later raises "AttributeError: 'NoneType' object
    # has no attribute 'is_pdf'". Numbers stay valid; re-fetch each one
    # fresh via out[n] once everything is built.
    all_pages = []  # list of (out_page_number, source_sheet_page)

    def _place(page, r, dest_rect):
        src_bbox = pymupdf.Rect(r["X0"], r["Y0"], r["X1"], r["Y1"])
        try:
            page.show_pdf_page(dest_rect, src, int(r["Page"]) - 1, clip=src_bbox)
        except Exception:
            pass

    if manual_dest is not None:
        for sheet_page, pairs in manual_dest.items():
            page = out.new_page(width=w, height=h)
            all_pages.append((page.number, sheet_page))
            for r, dest in pairs:
                if (r["X1"] - r["X0"]) <= 0 or (r["Y1"] - r["Y0"]) <= 0:
                    continue
                if dest["w"] <= 0 or dest["h"] <= 0:
                    continue
                dest_rect = pymupdf.Rect(dest["x"], dest["y"], dest["x"] + dest["w"], dest["y"] + dest["h"])
                _place(page, r, dest_rect)
    else:
        valid_regions = [r for r in selected_regions if (r["X1"] - r["X0"]) > 0 and (r["Y1"] - r["Y0"]) > 0]
        for sheet_regions in _group_by_sheet(valid_regions):
            # Squeeze this sheet's items onto a single output page:
            # binary-search a uniform scale small enough that a greedy
            # row-pack of the sheet's items fits within one page.
            scale = _best_single_page_scale(sheet_regions, content_w, content_h, spacing)
            rows, _ = _shelf_pack(sheet_regions, content_w, content_h, spacing, scale)
            page = out.new_page(width=w, height=h)
            all_pages.append((page.number, int(sheet_regions[0]["Page"])))
            cursor_y = margin
            for row_h, items in rows:
                cursor_x = margin
                for r, sw, sh in items:
                    dest_rect = pymupdf.Rect(cursor_x, cursor_y, cursor_x + sw, cursor_y + sh)
                    _place(page, r, dest_rect)
                    cursor_x += sw + spacing
                cursor_y += row_h + spacing

    for page_number, source_page in all_pages:
        page_footer = dict(footer_fields)
        if source_page is not None:
            per_sheet = _extract_title_block_fields_from_page(src[source_page - 1])
            # fall back to the global sidebar value if this sheet's own
            # title block couldn't be read (detection is best-effort)
            if per_sheet.get("drawing_name"):
                page_footer["drawing_name"] = per_sheet["drawing_name"]
            if per_sheet.get("drawing_no"):
                page_footer["drawing_no"] = per_sheet["drawing_no"]
        _draw_footer(out[page_number], w, h, margin, page_footer)

    buf = out.tobytes()
    out.close()
    src.close()
    return buf


# ============================================================
# Manual drag & resize layout editor (optional, needs
# streamlit-drawable-canvas)
# ============================================================
CANVAS_MAX_W = 720
CANVAS_MAX_H = 900


def _canvas_dims(page_w, page_h):
    cw = CANVAS_MAX_W
    ch = cw * page_h / page_w
    if ch > CANVAS_MAX_H:
        ch = CANVAS_MAX_H
        cw = ch * page_w / page_h
    return cw, ch


def build_canvas_initial_layout(selected_rows, page_w, page_h, margin, spacing):
    """Starting positions for the layout editor: reuse the squeeze-pack
    layout as a sensible default, converted into canvas pixel space."""
    cw, ch = _canvas_dims(page_w, page_h)
    canvas_scale = cw / page_w
    content_w = page_w - 2 * margin
    content_h = page_h - 2 * margin

    scale = _best_single_page_scale(selected_rows, content_w, content_h, spacing)
    rows, _ = _shelf_pack(selected_rows, content_w, content_h, spacing, scale)

    objects = []
    order = []
    cursor_y = margin
    for row_h, items in rows:
        cursor_x = margin
        for r, sw, sh in items:
            objects.append({
                "type": "rect", "left": cursor_x * canvas_scale, "top": cursor_y * canvas_scale,
                "width": sw * canvas_scale, "height": sh * canvas_scale,
                "scaleX": 1, "scaleY": 1, "angle": 0,
                "fill": "rgba(46, 96, 166, 0.15)", "stroke": "#2e60a6", "strokeWidth": 1.5,
                "lockRotation": True, "hasRotatingPoint": False,
            })
            order.append((int(r["Page"]), r["Kind"], r["Title"], r["X0"], r["Y0"], r["X1"], r["Y1"]))
            cursor_x += sw + spacing
        cursor_y += row_h + spacing
    return {"version": "4.4.0", "objects": objects}, order


def build_grid_initial_layout(selected_rows, page_w, page_h, margin, spacing, cols, rows):
    """Arranges selected regions into a fixed cols x rows grid, each item
    scaled to fit its cell (preserving its own aspect ratio, centered).
    If there are more items than cells, extra rows are added automatically
    so nothing gets dropped."""
    n = len(selected_rows)
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    if n > cols * rows:
        rows = -(-n // cols)  # ceil division -- grow rows to fit everyone

    cw, ch = _canvas_dims(page_w, page_h)
    canvas_scale = cw / page_w
    content_w = page_w - 2 * margin
    content_h = page_h - 2 * margin
    cell_w = (content_w - spacing * (cols - 1)) / cols
    cell_h = (content_h - spacing * (rows - 1)) / rows

    objects, order = [], []
    for i, r in enumerate(selected_rows):
        row_i, col_i = divmod(i, cols)
        cell_x = margin + col_i * (cell_w + spacing)
        cell_y = margin + row_i * (cell_h + spacing)

        rw, rh = r["X1"] - r["X0"], r["Y1"] - r["Y0"]
        if rw <= 0 or rh <= 0:
            continue
        item_scale = min(cell_w / rw, cell_h / rh)
        iw, ih = rw * item_scale, rh * item_scale
        ix = cell_x + (cell_w - iw) / 2  # center within the cell
        iy = cell_y + (cell_h - ih) / 2

        objects.append({
            "type": "rect", "left": ix * canvas_scale, "top": iy * canvas_scale,
            "width": iw * canvas_scale, "height": ih * canvas_scale,
            "scaleX": 1, "scaleY": 1, "angle": 0,
            "fill": "rgba(46, 96, 166, 0.15)", "stroke": "#2e60a6", "strokeWidth": 1.5,
            "lockRotation": True, "hasRotatingPoint": False,
        })
        order.append((int(r["Page"]), r["Kind"], r["Title"], r["X0"], r["Y0"], r["X1"], r["Y1"]))
    return {"version": "4.4.0", "objects": objects}, order


def canvas_objects_to_dest(json_objects, page_w, page_h):
    """Converts fabric.js canvas object dicts back into PDF-point
    {'x','y','w','h'} placements, in the same order they were passed in."""
    cw, ch = _canvas_dims(page_w, page_h)
    canvas_scale = cw / page_w
    dests = []
    for obj in json_objects:
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0) * obj.get("scaleX", 1)
        height = obj.get("height", 0) * obj.get("scaleY", 1)
        dests.append({
            "x": left / canvas_scale, "y": top / canvas_scale,
            "w": width / canvas_scale, "h": height / canvas_scale,
        })
    return dests


# ============================================================
# Sheet grouping (treat sheets with the exact same set of
# view/schedule names as one repeatable project)
# ============================================================
def _signature_key(signature):
    """Stable string key for a (Kind, Title) signature, for cache storage."""
    return "||".join(sorted(f"{k}:{t}" for k, t in signature))


def group_sheets_by_signature(regions_df):
    """Groups sheet page numbers by the exact set of (Kind, Title) detected
    on them -- sheets with the identical set of view/schedule names are
    the same "project" and get grouped together. Returns a list of group
    dicts (largest group first), each: {"id", "key", "signature", "pages"}."""
    by_page = {}
    for _, row in regions_df.iterrows():
        by_page.setdefault(int(row["Page"]), []).append((row["Kind"], row["Title"]))

    sig_to_pages = {}
    for page, items in by_page.items():
        sig = frozenset(items)
        sig_to_pages.setdefault(sig, []).append(page)

    groups = [{"signature": sig, "pages": sorted(pages)} for sig, pages in sig_to_pages.items()]
    groups.sort(key=lambda g: (-len(g["pages"]), sorted(g["signature"])))
    for i, g in enumerate(groups):
        g["id"] = i + 1
        g["key"] = _signature_key(g["signature"])
    return groups


def find_incomplete_sheets(groups):
    """Flags a smaller group whose signature is a proper subset of a
    bigger group's signature -- those sheets probably belong with the
    bigger group but are missing a view/schedule the rest of it has
    (not detected, or genuinely absent from just that sheet)."""
    flags = []
    for g in groups:
        for other in groups:
            if other is g or len(other["pages"]) <= len(g["pages"]):
                continue
            if g["signature"] < other["signature"]:  # proper subset
                missing = sorted(other["signature"] - g["signature"])
                flags.append({"pages": g["pages"], "closest_group_id": other["id"], "missing": missing})
                break  # groups is sorted biggest-first, so this is the best match
    return flags


def _format_page_ranges(pages):
    """[1, 2, 3, 5, 6, 9] -> '1-3, 5-6, 9'"""
    pages = sorted(pages)
    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append((start, prev))
        start = prev = p
    ranges.append((start, prev))
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


# ============================================================
# Page-range spec parsing (text box: "all", "2-71", "2-10, 13-15")
# ============================================================
def parse_page_range_spec(spec, n_pages):
    """Parses a page-range string into a sorted, de-duplicated list of valid
    page numbers (1-indexed). Accepts:
      - "all" (or blank) -> every page
      - a single range: "2-71"
      - a comma-separated mix of single pages and ranges: "2-10, 13-15, 20"
    Raises ValueError with a user-facing message on malformed input or
    out-of-range page numbers."""
    spec = (spec or "").strip()
    if not spec or spec.lower() == "all":
        return list(range(1, n_pages + 1))

    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not all(b.strip().isdigit() for b in bounds):
                raise ValueError(f"Couldn't parse range '{part}'. Use a format like 2-10.")
            a, b = int(bounds[0].strip()), int(bounds[1].strip())
            if a > b:
                a, b = b, a
            pages.update(range(a, b + 1))
        elif part.isdigit():
            pages.add(int(part))
        else:
            raise ValueError(f"Couldn't parse '{part}'. Use a page number or a range like 2-10.")

    out_of_range = sorted(p for p in pages if p < 1 or p > n_pages)
    if out_of_range:
        raise ValueError(
            f"Page(s) {', '.join(str(p) for p in out_of_range)} are out of range -- "
            f"this PDF only has {n_pages} page(s)."
        )
    return sorted(pages)


# ============================================================
# UI
# ============================================================
st.title("📐 Sheet Region Exporter")
st.caption(
    "Upload a drawing sheet PDF. Schedules, views, and the notes block are "
    "auto-detected on every sheet. Sheets sharing the exact same set of "
    "view/schedule names are grouped into one project -- check what you want "
    "once per group and it's applied to every sheet in it -- then export "
    "onto a new PDF (one output page per sheet) at whatever paper size you need."
)

with st.sidebar:
    st.header("Output Settings")
    paper_key = st.selectbox("Paper size", list(PAPER_SIZES_PT.keys()))
    orientation = st.radio("Orientation", ["Portrait", "Landscape"], horizontal=True)
    layout_options = ["Squeeze each sheet onto one page"]
    if HAS_CANVAS:
        layout_options.append("Manual layout (drag & resize)")
    layout_mode = st.radio(
        "Layout",
        layout_options,
        index=0,
        help="Each input sheet (page) is its own project and gets its own "
             "output page -- items from two different sheets are never "
             "combined onto the same output page. 'Squeeze each sheet onto "
             "one page' shrinks everything checked on a given sheet, as a "
             "group, to the largest uniform scale that still fits that "
             "sheet on a single output page. 'Manual layout' lets you drag "
             "and resize each item yourself before exporting, across the "
             "whole current selection on one page.",
    )
    manual_layout = layout_mode.startswith("Manual")
    if not HAS_CANVAS:
        st.caption("Tip: `pip install streamlit-drawable-canvas` to unlock a "
                   "drag-and-resize layout mode.")
    with st.expander("Advanced layout"):
        margin = st.number_input("Margin (pt)", min_value=0, max_value=144, value=36, step=6)
        spacing = st.number_input("Spacing between items (pt)", min_value=0, max_value=72, value=18, step=6)

uploaded = st.file_uploader("📄 Upload sheet PDF", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")

if uploaded is not None:
    pdf_bytes = uploaded.read()
    if st.session_state.pdf_bytes != pdf_bytes:
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_name = uploaded.name
        st.session_state.pdf_hash = _pdf_hash(pdf_bytes)
        st.session_state.detected = False
        st.session_state.regions_df = pd.DataFrame(
            columns=["Include", "Page", "Kind", "Title", "X0", "Y0", "X1", "Y1"]
        )

        cache = _load_cache()
        entry = cache.get(st.session_state.pdf_hash, {})
        saved_footer = entry.get("footer", {})
        if saved_footer:
            # We've seen this exact file before -- restore what was typed last time.
            st.session_state.drawing_name = saved_footer.get("drawing_name", "")
            st.session_state.drawing_no = saved_footer.get("drawing_no", "")
            st.session_state.project_name = saved_footer.get("project_name", "")
        else:
            guessed = extract_title_block_fields(pdf_bytes)
            st.session_state.drawing_name = guessed["drawing_name"]
            st.session_state.drawing_no = guessed["drawing_no"]
            st.session_state.project_name = guessed["project_name"]

if st.session_state.pdf_bytes is not None:
    try:
        with pdfplumber.open(io.BytesIO(st.session_state.pdf_bytes)) as _pdf:
            n_pages = len(_pdf.pages)
    except Exception as e:
        st.error(f"Couldn't open that PDF: {e}")
        n_pages = 0

    with st.sidebar:
        st.header("Footer")
        st.caption(
            "Printed along the bottom of every exported page. Project Name is shared "
            "across every sheet. Drawing Name and Drawing No. are normally read fresh "
            "from each sheet's own title block, so every output page gets its correct "
            "value automatically -- the fields below are only a fallback, used if a "
            "particular sheet's title block can't be read."
        )
        st.session_state.project_name = st.text_input("Project Name", st.session_state.project_name)
        st.session_state.drawing_name = st.text_input(
            "Drawing Name (fallback)", st.session_state.drawing_name)
        st.session_state.drawing_no = st.text_input(
            "Drawing No. (fallback)", st.session_state.drawing_no)


def _persist_current_state():
    """Saves checkbox states + footer field values for the current PDF so
    they're recalled next time this same file is opened."""
    if not st.session_state.get("pdf_hash"):
        return
    cache = _load_cache()
    entry = cache.setdefault(st.session_state.pdf_hash, {})
    entry["footer"] = {
        "drawing_name": st.session_state.drawing_name,
        "drawing_no": st.session_state.drawing_no,
        "project_name": st.session_state.project_name,
    }
    regions_map = {}
    for _, row in st.session_state.regions_df.iterrows():
        key = _region_key(int(row["Page"]), row["Kind"], row["Title"])
        regions_map[key] = bool(row["Include"])
    entry["regions"] = regions_map
    cache[st.session_state.pdf_hash] = entry
    _save_cache(cache)


if st.session_state.pdf_bytes is not None:
    _persist_current_state()

    if n_pages:
        if n_pages == 1:
            pages_to_scan = [1]
        else:
            page_spec = st.text_input(
                "Pages to scan",
                value="all",
                help="Type 'all' to scan every page, a single range like 2-71, "
                     "or a comma-separated mix of pages/ranges like 2-10, 13-15.",
            )
            try:
                pages_to_scan = parse_page_range_spec(page_spec, n_pages)
                if not pages_to_scan:
                    st.warning("No pages selected -- type 'all' or a range like 2-71.")
                else:
                    st.caption(f"Will scan {len(pages_to_scan)} page(s): "
                               f"{_format_page_ranges(pages_to_scan)}")
            except ValueError as e:
                st.error(str(e))
                pages_to_scan = []

        force_rescan = st.checkbox(
            "Force re-scan (ignore cached results from a previous Detect run)",
            value=False,
            help="Detection results are cached per page for this file. Leave this unchecked "
                 "to re-run Detect quickly after only changing the page range above -- "
                 "already-scanned pages are reused instead of re-processed. Check this if "
                 "you've edited the PDF itself, or want to redo detection from scratch.",
        )

        if st.button("🔍 Detect Schedules / Views / Notes", type="primary", disabled=not pages_to_scan):
            progress_bar = st.progress(0.0, text=f"Scanning 0/{len(pages_to_scan)} sheet(s)...")

            def _report_progress(done, total):
                progress_bar.progress(done / total, text=f"Scanning {done}/{total} sheet(s)...")

            try:
                cache = _load_cache()
                entry = cache.get(st.session_state.pdf_hash, {})
                page_cache = entry.get("page_regions", {})
                n_cached_before = sum(1 for pno in pages_to_scan if str(pno) in page_cache) if not force_rescan else 0

                regions, page_errors, updated_page_cache = detect_all_pages(
                    st.session_state.pdf_bytes, pages_to_scan,
                    on_progress=_report_progress, page_cache=page_cache, force_rescan=force_rescan,
                )
                progress_bar.empty()

                # persist the (now larger) per-page detection cache
                cache = _load_cache()
                entry = cache.setdefault(st.session_state.pdf_hash, {})
                entry["page_regions"] = updated_page_cache
                cache[st.session_state.pdf_hash] = entry
                _save_cache(cache)

                saved_includes = entry.get("regions", {})
                rows = [{
                    "Include": bool(saved_includes.get(
                        _region_key(r["page"] + 1, r["kind"], r["title"]), False)),
                    "Page": r["page"] + 1, "Kind": r["kind"], "Title": r["title"],
                    "X0": round(r["bbox"][0], 1), "Y0": round(r["bbox"][1], 1),
                    "X1": round(r["bbox"][2], 1), "Y1": round(r["bbox"][3], 1),
                } for r in regions]
                # pages now finish scanning in whatever order threads complete
                # them, not the order they were requested in -- restore
                # reading order (page, then top-to-bottom, then left-to-right)
                rows.sort(key=lambda r: (r["Page"], round(r["Y0"] / 40), r["X0"]))
                st.session_state.regions_df = pd.DataFrame(rows)
                st.session_state.detected = True
                if page_errors:
                    bad_pages = ", ".join(str(p) for p, _ in page_errors)
                    st.warning(f"{len(page_errors)} sheet(s) failed to scan and were skipped "
                               f"(page {bad_pages}). Every other sheet's results below are still "
                               "usable -- add regions manually for the skipped ones if needed.")
                if not rows:
                    st.warning("No schedules, views, or notes were detected on the selected page(s). "
                               "You can still add regions manually in the table below.")
                else:
                    n_restored = sum(1 for r in rows if r["Include"])
                    msg = f"Found {len(rows)} region(s)"
                    if n_cached_before:
                        msg += f" -- reused cached results for {n_cached_before} already-scanned page(s)"
                    msg += ", all unchecked by default." if not n_restored else "."
                    if n_restored:
                        msg += (f" Restored {n_restored} checkmark(s) you'd saved for this "
                                "file previously.")
                    st.success(msg)
            except Exception as e:
                progress_bar.empty()
                st.error(f"Detection failed: {type(e).__name__}: {e}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

        if st.session_state.detected or not st.session_state.regions_df.empty:
            st.subheader("Preview")
            preview_page = st.selectbox(
                "Page to preview", pages_to_scan, format_func=lambda p: f"Page {p}"
            ) if len(pages_to_scan) > 1 else pages_to_scan[0]

            df_now = st.session_state.regions_df
            page_regions = []
            for _, row in df_now.iterrows():
                if int(row["Page"]) == preview_page:
                    page_regions.append({
                        "kind": row["Kind"], "title": row["Title"],
                        "bbox": (row["X0"], row["Y0"], row["X1"], row["Y1"]),
                    })
            try:
                img = render_preview_image(st.session_state.pdf_bytes, preview_page - 1, page_regions)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Couldn't render preview: {e}")

            st.subheader("Sheet Groups")
            st.caption(
                "Sheets with the exact same set of view/schedule names are treated as one "
                "repeatable project: check the ones you want from a group once below, and "
                "that selection is applied to every sheet in that group automatically."
            )

            groups = group_sheets_by_signature(st.session_state.regions_df)
            incomplete_flags = find_incomplete_sheets(groups)
            for f in incomplete_flags:
                missing_str = ", ".join(f"{k}: {t}" for k, t in f["missing"])
                st.warning(
                    f"Sheet(s) {_format_page_ranges(f['pages'])} look like they belong with "
                    f"Group {f['closest_group_id']} but are missing: {missing_str}. They'll "
                    "only export whatever was actually detected on them."
                )

            group_cache = _load_cache()
            group_cache_entry = group_cache.setdefault(st.session_state.pdf_hash, {})
            group_selections = group_cache_entry.setdefault("group_selections", {})

            for g in groups:
                sig_sorted = sorted(g["signature"])
                saved = {tuple(x) for x in group_selections.get(g["key"], [])}
                with st.container(border=True):
                    st.markdown(f"**Group {g['id']}** -- {len(g['pages'])} sheet(s) "
                                f"(page {_format_page_ranges(g['pages'])})")
                    n_cols = min(len(sig_sorted), 4) or 1
                    cols = st.columns(n_cols)
                    checked = set()
                    for i, (kind, title) in enumerate(sig_sorted):
                        default = (kind, title) in saved
                        val = cols[i % n_cols].checkbox(
                            f"{kind}: {title}", value=default, key=f"grp_{g['key']}_{kind}_{title}"
                        )
                        if val:
                            checked.add((kind, title))

                df = st.session_state.regions_df
                mask = df["Page"].astype(int).isin(g["pages"])
                for kind, title in sig_sorted:
                    row_mask = mask & (df["Kind"] == kind) & (df["Title"] == title)
                    df.loc[row_mask, "Include"] = (kind, title) in checked
                st.session_state.regions_df = df
                group_selections[g["key"]] = [list(t) for t in checked]

            group_cache[st.session_state.pdf_hash] = group_cache_entry
            _save_cache(group_cache)

            st.divider()

            with st.expander("Fine-tune individual detections (advanced)"):
                st.caption(
                    "Per-sheet detail, for the rare case a detection box or title needs a manual "
                    "correction. Titles and box coordinates (in PDF points, top-left origin) are "
                    "editable, and you can add or delete rows. Note: the Include checkbox here is "
                    "normally driven by the Sheet Groups selections above -- toggling it manually is "
                    "only a temporary override and will be replaced the next time a group checkbox "
                    "changes or a Select All/Deselect All button runs."
                )
                colored_legend = " &nbsp; ".join(
                    f'<span style="color:rgb{c}">■</span> {k}' for k, c in KIND_COLORS.items()
                )
                st.markdown(colored_legend, unsafe_allow_html=True)

                edited = st.data_editor(
                    st.session_state.regions_df,
                    column_config={
                        "Include": st.column_config.CheckboxColumn("Include", default=False),
                        "Page": st.column_config.NumberColumn("Page", min_value=1, max_value=n_pages, step=1),
                        "Kind": st.column_config.SelectboxColumn("Kind", options=["Schedule", "View", "Notes"]),
                        "Title": st.column_config.TextColumn("Title", width="medium"),
                        "X0": st.column_config.NumberColumn("X0", format="%.1f"),
                        "Y0": st.column_config.NumberColumn("Y0", format="%.1f"),
                        "X1": st.column_config.NumberColumn("X1", format="%.1f"),
                        "Y1": st.column_config.NumberColumn("Y1", format="%.1f"),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="editor_regions",
                )
                st.session_state.regions_df = edited
                _persist_current_state()

                col_a, col_b, col_c, _ = st.columns([1, 1, 1.6, 2.4])
                with col_a:
                    if st.button("Select All"):
                        st.session_state.regions_df["Include"] = True
                        _persist_current_state()
                        st.rerun()
                with col_b:
                    if st.button("Deselect All"):
                        st.session_state.regions_df["Include"] = False
                        _persist_current_state()
                        st.rerun()
                with col_c:
                    if st.button("Forget saved checks for this file"):
                        cache = _load_cache()
                        cache.pop(st.session_state.pdf_hash, None)
                        _save_cache(cache)
                        st.session_state.regions_df["Include"] = False
                        st.rerun()

            st.divider()
            selected = st.session_state.regions_df[st.session_state.regions_df["Include"] == True]
            st.write(f"**{len(selected)}** region(s) selected for export.")
            selected_sorted = selected.sort_values(by=["Page", "Y0", "X0"])
            selected_rows = selected_sorted.to_dict("records")

            manual_dest_list = None
            if manual_layout and not selected.empty:
                st.subheader("Layout Editor")
                st.caption(
                    "Design the layout once, per named view/schedule -- drag an item to "
                    "reposition it and drag a corner handle to resize it (hold Shift to keep "
                    "proportions), or pick a column x row grid below and click Apply. This same "
                    f"layout is then repeated as its own {paper_key.split(' ')[0]} "
                    f"{orientation.lower()} output page for every sheet that has these items -- "
                    "the light grey guide shows your margin."
                )

                page_w, page_h = PAPER_SIZES_PT[paper_key]
                if orientation == "Landscape":
                    page_w, page_h = page_h, page_w

                template_rows = _dedupe_rows_by_name(selected_rows)

                selection_key = tuple((r["Kind"], r["Title"]) for r in template_rows)
                if st.session_state.canvas_selection_key != selection_key:
                    initial, order = build_canvas_initial_layout(template_rows, page_w, page_h, margin, spacing)
                    st.session_state.canvas_initial = initial
                    st.session_state.canvas_region_order = order
                    st.session_state.canvas_selection_key = selection_key

                arrange_col1, arrange_col2, arrange_col3, arrange_col4 = st.columns([1.3, 0.8, 0.8, 1.1])
                with arrange_col1:
                    if st.button("↺ Reset to auto-packed"):
                        initial, order = build_canvas_initial_layout(template_rows, page_w, page_h, margin, spacing)
                        st.session_state.canvas_initial = initial
                        st.session_state.canvas_region_order = order
                with arrange_col2:
                    grid_cols = st.number_input("Columns", min_value=1, max_value=8, value=2, step=1, key="grid_cols")
                with arrange_col3:
                    grid_rows = st.number_input("Rows", min_value=1, max_value=8, value=1, step=1, key="grid_rows")
                with arrange_col4:
                    st.write("")  # vertical spacer to align button with the number inputs
                    if st.button(f"⊞ Apply {int(grid_cols)}x{int(grid_rows)} grid"):
                        initial, order = build_grid_initial_layout(
                            template_rows, page_w, page_h, margin, spacing, grid_cols, grid_rows)
                        st.session_state.canvas_initial = initial
                        st.session_state.canvas_region_order = order
                        if len(template_rows) > grid_cols * grid_rows:
                            st.info(f"{len(template_rows)} named item(s) don't fit a "
                                    f"{int(grid_cols)}x{int(grid_rows)} grid -- added extra row(s) "
                                    "so nothing was left out.")

                cw, ch = _canvas_dims(page_w, page_h)
                canvas_scale = cw / page_w
                margin_guide = Image.new("RGB", (int(cw), int(ch)), "white")
                gd = ImageDraw.Draw(margin_guide)
                gd.rectangle(
                    (margin * canvas_scale, margin * canvas_scale,
                     cw - margin * canvas_scale, ch - margin * canvas_scale),
                    outline=(210, 210, 210),
                )

                canvas_result = st_canvas(
                    fill_color="rgba(46, 96, 166, 0.15)",
                    stroke_width=2,
                    stroke_color="#2e60a6",
                    background_image=margin_guide,
                    update_streamlit=True,
                    height=int(ch), width=int(cw),
                    drawing_mode="transform",
                    initial_drawing=st.session_state.canvas_initial,
                    key="layout_canvas",
                )

                if canvas_result.json_data is not None:
                    objs = canvas_result.json_data.get("objects", [])
                    if len(objs) == len(st.session_state.canvas_region_order):
                        manual_dest_list = canvas_objects_to_dest(objs, page_w, page_h)
                    else:
                        st.warning("Layout editor is out of sync with the current selection -- "
                                   "click 'Reset layout to auto-packed' above.")

            gen_disabled = selected.empty or (manual_layout and manual_dest_list is None)
            if st.button("📤 Generate PDF", type="primary", use_container_width=True, disabled=gen_disabled):
                with st.spinner("Building output PDF..."):
                    try:
                        footer_fields = {
                            "project_name": st.session_state.project_name.strip(),
                            "drawing_name": st.session_state.drawing_name.strip(),
                            "drawing_no": st.session_state.drawing_no.strip(),
                        }
                        if manual_layout and manual_dest_list is not None:
                            # manual_dest_list has one dest rect per distinct
                            # (Kind, Title) name, in canvas_region_order. Repeat
                            # that same template as one output page per sheet,
                            # each sheet contributing its own region for
                            # whichever names it actually has.
                            dest_by_name = {
                                (o[1], o[2]): dest
                                for o, dest in zip(st.session_state.canvas_region_order, manual_dest_list)
                            }
                            manual_dest_by_sheet = {}
                            for sheet_regions in _group_by_sheet(selected_rows):
                                sheet_page = int(sheet_regions[0]["Page"])
                                pairs = [
                                    (r, dest_by_name[(r["Kind"], r["Title"])])
                                    for r in sheet_regions
                                    if (r["Kind"], r["Title"]) in dest_by_name
                                ]
                                if pairs:
                                    manual_dest_by_sheet[sheet_page] = pairs
                            out_bytes = build_output_pdf(
                                st.session_state.pdf_bytes, None,
                                paper_key, orientation, margin=margin, spacing=spacing,
                                footer_fields=footer_fields, manual_dest=manual_dest_by_sheet,
                            )
                        else:
                            out_bytes = build_output_pdf(
                                st.session_state.pdf_bytes, selected_rows,
                                paper_key, orientation, margin=margin, spacing=spacing,
                                footer_fields=footer_fields,
                            )
                        base = os.path.splitext(st.session_state.pdf_name or "sheet")[0]
                        mode_tag = "MANUAL" if manual_layout else "SQUEEZE"
                        out_name = f"{base}_SELECTED_{paper_key.split(' ')[0]}_{orientation}_{mode_tag}.pdf"
                        with pymupdf.open(stream=out_bytes, filetype="pdf") as _chk:
                            n_out_pages = len(_chk)
                        st.success(f"Done! ({n_out_pages} page{'s' if n_out_pages != 1 else ''})")
                        st.download_button(
                            "⬇️ Download Exported PDF", data=out_bytes,
                            file_name=out_name, mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Export failed: {type(e).__name__}: {e}")
                        with st.expander("Details"):
                            st.code(traceback.format_exc())
else:
    st.info("Upload a PDF to get started.")
