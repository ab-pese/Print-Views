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

import io
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
    st.session_state.uploader_key = 0

st.set_page_config(page_title="Sheet Region Exporter", layout="wide")

# ============================================================
# Constants / tunables
# ============================================================
PAPER_SIZES_PT = {
    "Letter (8.5 x 11)": (612, 792),
    "Tabloid (11 x 17)": (792, 1224),
}
REGION_PAD = 8            # pt padding added to every finalized bbox
SCALE_RE = re.compile(r"=\s*\d+['\u2019]")
VIEW_KEYWORDS_RE = re.compile(
    r"\b(PLAN|SECTION|ELEVATION|DETAIL|ORTHO|3D|ISOMETRIC|VIEW|PERSPECTIVE)\b",
    re.IGNORECASE,
)
CAPTION_BLACKLIST = {
    "SEE PLAN", "SEE DETAIL", "SEE SECTION", "SEE ELEVATION",
    "TYP", "TYPICAL", "NOTES:",
}
TITLE_BLOCK_KW_RE = re.compile(
    r"DESIGN BY|DRAWN BY|CHECKED BY|REVISIONS|SHEET NO|PROJECT NO|"
    r"DRAWING NO|SCALE:|DATE:",
    re.IGNORECASE,
)
KIND_COLORS = {
    "Schedule": (46, 96, 166),   # blue
    "View": (95, 158, 64),       # green
    "Notes": (196, 128, 30),     # orange
}

# ============================================================
# Detection helpers
# ============================================================
def _group_words_into_segments(words, y_tol=2, x_gap=40):
    """Group words into visual text lines (by y), then split each line into
    separate segments wherever there's a big horizontal gap -- this keeps
    side-by-side captions (e.g. two views on the same row) distinct."""
    rows = {}
    for w in words:
        key = round(w["top"] / y_tol) * y_tol
        rows.setdefault(key, []).append(w)

    segments = []
    for k in sorted(rows.keys()):
        ws = sorted(rows[k], key=lambda w: w["x0"])
        cur = [ws[0]]
        for w in ws[1:]:
            if w["x0"] - cur[-1]["x1"] > x_gap:
                segments.append(cur)
                cur = [w]
            else:
                cur.append(w)
        segments.append(cur)

    out = []
    for seg in segments:
        text = " ".join(w["text"] for w in seg)
        bbox = (
            min(w["x0"] for w in seg), min(w["top"] for w in seg),
            max(w["x1"] for w in seg), max(w["bottom"] for w in seg),
        )
        out.append((text, bbox))
    return out


def _extend_bbox_to_trailing_rows(page_pl, bbox, max_gap=18, max_iter=6):
    """Some Revit-exported schedules have a final data row with no ruling
    lines below it, so pdfplumber's ruled-table detection clips it off.
    Walk downward picking up any word lines that are clearly a continuation
    (same x-range, small vertical gap) of the table."""
    x0, y0, x1, y1 = bbox
    words = page_pl.extract_words()
    for _ in range(max_iter):
        cand = [w for w in words
                if 0 <= (w["top"] - y1) <= max_gap
                and w["x0"] >= x0 - 5 and w["x1"] <= x1 + 5]
        if not cand:
            break
        new_y1 = max(w["bottom"] for w in cand)
        if new_y1 <= y1 + 0.5:
            break
        y1 = new_y1
    return (x0, y0, x1, y1)


def _detect_schedules(page_pl, page_area):
    """Detect ruled tables that look like real schedules (not stray
    dimension-line crossings or the title block)."""
    schedules = []
    try:
        tables_raw = page_pl.find_tables(
            table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
        )
    except Exception:
        tables_raw = []

    for t in tables_raw:
        try:
            data = t.extract()
        except Exception:
            continue
        rows = len(data)
        cols = max((len(r) for r in data), default=0)
        if rows < 2 or cols < 2:
            continue
        nonempty = sum(1 for r in data for c in r if c and str(c).strip())
        density = nonempty / (rows * cols)
        if density < 0.45:
            continue
        x0, y0, x1, y1 = t.bbox
        area = (x1 - x0) * (y1 - y0)
        if area > 0.5 * page_area:
            continue

        full_text = " ".join(str(c) for r in data for c in r if c)
        if len(TITLE_BLOCK_KW_RE.findall(full_text)) >= 2:
            continue  # this is the title block, not a schedule

        title = None
        if data and data[0][0] and all((c is None or not str(c).strip()) for c in data[0][1:]):
            title = str(data[0][0]).replace("\n", " ").strip()
        extended_bbox = _extend_bbox_to_trailing_rows(page_pl, (x0, y0, x1, y1))
        schedules.append({"bbox": extended_bbox, "title": title or "Schedule"})

    return schedules


def _detect_notes(page_pl, schedules, page_h):
    words = page_pl.extract_words()
    notes_top = notes_x0 = None
    for w in words:
        if w["text"].upper().startswith("NOTE"):
            notes_top, notes_x0 = w["top"], w["x0"]
            break
    if notes_top is None:
        return None

    cand = [w for w in words if w["top"] >= notes_top - 1 and notes_x0 - 15 <= w["x0"] <= notes_x0 + 450]
    lower_schedule_tops = [s["bbox"][1] for s in schedules if s["bbox"][1] > notes_top]
    limit_top = min(lower_schedule_tops) if lower_schedule_tops else page_h
    cand = [w for w in cand if w["top"] < limit_top - 2]
    if not cand:
        return None
    return (
        min(w["x0"] for w in cand), min(w["top"] for w in cand),
        max(w["x1"] for w in cand), max(w["bottom"] for w in cand),
    )


def _detect_view_captions(page_pl, schedules, notes_bbox):
    words = page_pl.extract_words()
    segs = _group_words_into_segments(words)
    scale_segs = [s for s in segs if SCALE_RE.search(s[0])]
    other_segs = [s for s in segs if not SCALE_RE.search(s[0])]

    views, used = [], set()
    # Pass 1: title line immediately above a scale callout (most reliable)
    for _, bbox_s in scale_segs:
        best, best_gap = None, 999
        for idx, (text_o, bbox_o) in enumerate(other_segs):
            if idx in used or text_o.strip().upper() in CAPTION_BLACKLIST:
                continue
            gap = bbox_s[1] - bbox_o[3]
            if 0 <= gap <= 25 and abs(bbox_o[0] - bbox_s[0]) < 30:
                if gap < best_gap:
                    best_gap, best = gap, idx
        if best is not None:
            text_o, bbox_o = other_segs[best]
            used.add(best)
            views.append({
                "title": text_o.strip(),
                "caption_bbox": (min(bbox_o[0], bbox_s[0]), bbox_o[1],
                                 max(bbox_o[2], bbox_s[2]), bbox_s[3]),
            })

    # Pass 2: unscaled views (e.g. 3D/isometric) identified by keyword only
    for idx, (text_o, bbox_o) in enumerate(other_segs):
        if idx in used:
            continue
        tu = text_o.strip().upper()
        if tu in CAPTION_BLACKLIST or tu.startswith("SEE "):
            continue
        if notes_bbox and bbox_o[1] >= notes_bbox[1] - 2 and bbox_o[1] <= notes_bbox[3] + 2:
            continue
        if any(abs(bbox_o[0] - s["bbox"][0]) < 5 and abs(bbox_o[1] - s["bbox"][1]) < 40 for s in schedules):
            continue
        if VIEW_KEYWORDS_RE.search(text_o) and len(text_o.split()) <= 4:
            views.append({"title": text_o.strip(), "caption_bbox": bbox_o})

    return views


def _find_drawing_blobs(page_mu, page_area, exclude_bboxes):
    """Rasterize the page and connected-component it to find blobs of vector
    drawing content (the actual view graphics), excluding schedule/notes
    regions and the page border frame."""
    dpi = 50
    zoom = dpi / 72
    pix = page_mu.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    mask = arr < 250

    def whiten(bbox, pad=3):
        x0, y0, x1, y1 = bbox
        px0 = max(int((x0 - pad) * zoom), 0)
        py0 = max(int((y0 - pad) * zoom), 0)
        px1 = min(int((x1 + pad) * zoom), mask.shape[1])
        py1 = min(int((y1 + pad) * zoom), mask.shape[0])
        mask[py0:py1, px0:px1] = False

    for b in exclude_bboxes:
        whiten(b)

    dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=1)
    labeled, _ = ndimage.label(dilated)

    comps = []
    for sl in ndimage.find_objects(labeled):
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        area_px = (x1 - x0) * (y1 - y0)
        if area_px < 15:
            continue
        density = mask[sl].sum() / area_px
        px0, py0, px1, py1 = x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom
        bbox_area = (px1 - px0) * (py1 - py0)
        if bbox_area > 0.25 * page_area and density < 0.08:
            continue  # sheet border frame
        comps.append((px0, py0, px1, py1))

    # merge nearby components into blobs
    pad = 20

    def expand(b, p):
        return (b[0] - p, b[1] - p, b[2] + p, b[3] + p)

    def overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    parent = list(range(len(comps)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if overlaps(expand(comps[i], pad), expand(comps[j], pad)):
                union(i, j)

    groups = {}
    for i, b in enumerate(comps):
        groups.setdefault(find(i), []).append(b)

    blobs = []
    for g in groups.values():
        blobs.append((
            min(b[0] for b in g), min(b[1] for b in g),
            max(b[2] for b in g), max(b[3] for b in g),
        ))
    return blobs


def detect_regions_on_page(page_pl, page_mu, page_index):
    """Returns a list of region dicts for one page: kind/title/bbox."""
    page_area = page_pl.width * page_pl.height

    schedules = _detect_schedules(page_pl, page_area)
    notes_bbox = _detect_notes(page_pl, schedules, page_pl.height)
    views = _detect_view_captions(page_pl, schedules, notes_bbox)

    exclude = [s["bbox"] for s in schedules]
    if notes_bbox:
        exclude.append(notes_bbox)
    blobs = _find_drawing_blobs(page_mu, page_area, exclude)

    def expand(b, p):
        return (b[0] - p, b[1] - p, b[2] + p, b[3] + p)

    def overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    regions = []
    for s in schedules:
        x0, y0, x1, y1 = s["bbox"]
        regions.append({
            "page": page_index, "kind": "Schedule", "title": s["title"],
            "bbox": (x0 - REGION_PAD, y0 - REGION_PAD, x1 + REGION_PAD, y1 + REGION_PAD),
        })

    if notes_bbox:
        x0, y0, x1, y1 = notes_bbox
        regions.append({
            "page": page_index, "kind": "Notes", "title": "Notes",
            "bbox": (x0 - REGION_PAD, y0 - REGION_PAD, x1 + REGION_PAD, y1 + REGION_PAD),
        })

    for v in views:
        cap = v["caption_bbox"]
        candidates = [b for b in blobs if overlaps(expand(cap, 10), b)]

        # also look for the nearest blob sitting above the caption with
        # x-range overlap, even across a visible gap (e.g. isometric views
        # where the caption sits well below the actual drawing) -- union
        # with any direct-overlap match above so both the caption text and
        # the real drawing content are captured
        best_gap, best_far = 1e9, None
        for b in blobs:
            x_overlap = min(cap[2], b[2]) - max(cap[0], b[0])
            gap = cap[1] - b[3]
            if x_overlap > 0 and 0 <= gap <= 300 and gap < best_gap:
                best_gap, best_far = gap, b
        if best_far is not None:
            candidates.append(best_far)

        if candidates:
            full = (
                min(min(c[0] for c in candidates), cap[0]),
                min(min(c[1] for c in candidates), cap[1]),
                max(max(c[2] for c in candidates), cap[2]),
                max(max(c[3] for c in candidates), cap[3]),
            )
        else:
            # nothing matched -- fall back to a generous guess above the
            # caption; the user should double-check/adjust this one
            full = (cap[0] - 100, max(cap[1] - 220, 0), cap[2] + 100, cap[3])
        x0, y0, x1, y1 = full
        regions.append({
            "page": page_index, "kind": "View", "title": v["title"],
            "bbox": (x0 - REGION_PAD, y0 - REGION_PAD, x1 + REGION_PAD, y1 + REGION_PAD),
        })

    # reading order: top-to-bottom, then left-to-right
    regions.sort(key=lambda r: (round(r["bbox"][1] / 40), r["bbox"][0]))
    return regions


def detect_all_pages(pdf_bytes, page_numbers):
    """page_numbers is a 1-indexed list of pages to scan."""
    all_regions = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf_pl:
        doc_mu = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        for pno in page_numbers:
            idx = pno - 1
            page_pl = pdf_pl.pages[idx]
            page_mu = doc_mu[idx]
            all_regions.extend(detect_regions_on_page(page_pl, page_mu, idx))
        doc_mu.close()
    return all_regions


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
def build_output_pdf(pdf_bytes, selected_regions, paper_key, orientation,
                      margin=36, spacing=18, title_fontsize=11):
    w, h = PAPER_SIZES_PT[paper_key]
    if orientation == "Landscape":
        w, h = h, w

    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()
    content_w = w - 2 * margin
    content_h = h - 2 * margin

    page = out.new_page(width=w, height=h)
    cursor_y = margin
    title_h = title_fontsize + 8

    for r in selected_regions:
        x0, y0, x1, y1 = r["X0"], r["Y0"], r["X1"], r["Y1"]
        rw, rh = x1 - x0, y1 - y0
        if rw <= 0 or rh <= 0:
            continue
        src_bbox = pymupdf.Rect(x0, y0, x1, y1)

        scale = content_w / rw
        scaled_h = rh * scale
        needed = title_h + scaled_h
        if needed > content_h:
            scale2 = (content_h - title_h) / rh
            scale = min(scale, scale2)
            scaled_h = rh * scale
            needed = title_h + scaled_h

        fits_full_size = (title_h + rh * (content_w / rw)) <= content_h + 1e-6
        remaining = margin + content_h - cursor_y
        if cursor_y > margin and (
            needed > remaining or (fits_full_size and needed > remaining)
        ):
            page = out.new_page(width=w, height=h)
            cursor_y = margin
            # recompute scale fresh for the new page (in case it was shrunk to fit before)
            scale = content_w / rw
            scaled_h = rh * scale
            needed = title_h + scaled_h
            if needed > content_h:
                scale2 = (content_h - title_h) / rh
                scale = min(scale, scale2)
                scaled_h = rh * scale
                needed = title_h + scaled_h

        page.insert_text((margin, cursor_y + title_fontsize),
                          f"{r['Kind']}: {r['Title']}", fontsize=title_fontsize, fontname="helv")
        top = cursor_y + title_h
        dest_rect = pymupdf.Rect(margin, top, margin + rw * scale, top + scaled_h)
        try:
            page.show_pdf_page(dest_rect, src, int(r["Page"]) - 1, clip=src_bbox)
        except Exception:
            continue
        page.draw_rect(dest_rect, color=(0.6, 0.6, 0.6), width=0.5)
        cursor_y = top + scaled_h + spacing

    buf = out.tobytes()
    out.close()
    src.close()
    return buf


# ============================================================
# UI
# ============================================================
st.title("📐 Sheet Region Exporter")
st.caption(
    "Upload a drawing sheet PDF. Schedules, views, and the notes block are "
    "auto-detected and listed below in reading order -- check the ones you "
    "want and export them onto a new PDF at whatever paper size you need."
)

with st.sidebar:
    st.header("Output Settings")
    paper_key = st.selectbox("Paper size", list(PAPER_SIZES_PT.keys()))
    orientation = st.radio("Orientation", ["Portrait", "Landscape"], horizontal=True)
    with st.expander("Advanced layout"):
        margin = st.number_input("Margin (pt)", min_value=0, max_value=144, value=36, step=6)
        spacing = st.number_input("Spacing between items (pt)", min_value=0, max_value=72, value=18, step=6)

uploaded = st.file_uploader("📄 Upload sheet PDF", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")

if uploaded is not None:
    pdf_bytes = uploaded.read()
    if st.session_state.pdf_bytes != pdf_bytes:
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_name = uploaded.name
        st.session_state.detected = False
        st.session_state.regions_df = pd.DataFrame(
            columns=["Include", "Page", "Kind", "Title", "X0", "Y0", "X1", "Y1"]
        )

if st.session_state.pdf_bytes is not None:
    try:
        with pdfplumber.open(io.BytesIO(st.session_state.pdf_bytes)) as _pdf:
            n_pages = len(_pdf.pages)
    except Exception as e:
        st.error(f"Couldn't open that PDF: {e}")
        n_pages = 0

    if n_pages:
        page_choices = list(range(1, n_pages + 1))
        pages_to_scan = page_choices if n_pages == 1 else st.multiselect(
            "Pages to scan", page_choices, default=page_choices
        )

        if st.button("🔍 Detect Schedules / Views / Notes", type="primary"):
            with st.spinner("Scanning sheet layout..."):
                try:
                    regions = detect_all_pages(st.session_state.pdf_bytes, pages_to_scan)
                    rows = [{
                        "Include": True, "Page": r["page"] + 1, "Kind": r["kind"], "Title": r["title"],
                        "X0": round(r["bbox"][0], 1), "Y0": round(r["bbox"][1], 1),
                        "X1": round(r["bbox"][2], 1), "Y1": round(r["bbox"][3], 1),
                    } for r in regions]
                    st.session_state.regions_df = pd.DataFrame(rows)
                    st.session_state.detected = True
                    if not rows:
                        st.warning("No schedules, views, or notes were detected on the selected page(s). "
                                   "You can still add regions manually in the table below.")
                    else:
                        st.success(f"Found {len(rows)} region(s). Review the boxes below, adjust as needed.")
                except Exception as e:
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

            st.subheader("Detected Regions")
            st.caption(
                "Uncheck anything you don't want exported. Titles and box coordinates (in PDF points, "
                "top-left origin) are editable if a detection looks off -- you can also add or delete rows."
            )
            colored_legend = " &nbsp; ".join(
                f'<span style="color:rgb{c}">■</span> {k}' for k, c in KIND_COLORS.items()
            )
            st.markdown(colored_legend, unsafe_allow_html=True)

            edited = st.data_editor(
                st.session_state.regions_df,
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include", default=True),
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

            col_a, col_b, _ = st.columns([1, 1, 3])
            with col_a:
                if st.button("Select All"):
                    st.session_state.regions_df["Include"] = True
                    st.rerun()
            with col_b:
                if st.button("Deselect All"):
                    st.session_state.regions_df["Include"] = False
                    st.rerun()

            st.divider()
            selected = st.session_state.regions_df[st.session_state.regions_df["Include"] == True]
            st.write(f"**{len(selected)}** region(s) selected for export.")

            if st.button("📤 Generate PDF", type="primary", use_container_width=True, disabled=selected.empty):
                with st.spinner("Building output PDF..."):
                    try:
                        selected_sorted = selected.sort_values(by=["Page", "Y0", "X0"])
                        out_bytes = build_output_pdf(
                            st.session_state.pdf_bytes,
                            selected_sorted.to_dict("records"),
                            paper_key, orientation, margin=margin, spacing=spacing,
                        )
                        base = os.path.splitext(st.session_state.pdf_name or "sheet")[0]
                        out_name = f"{base}_SELECTED_{paper_key.split(' ')[0]}_{orientation}.pdf"
                        st.success("Done!")
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
