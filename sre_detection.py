"""
sre_detection
=============
Standalone, side-effect-free page-detection engine for Sheet Region
Exporter. Deliberately kept in its own module (no Streamlit imports, no
top-level app code) so it can be safely imported inside worker
*processes*: multiprocessing (used for real parallelism across pages,
since pdfplumber's parsing is pure Python and doesn't release the GIL --
threads can't run it concurrently, only separate processes can) needs to
pickle a reference to whatever function each task calls. If that function
lived inside the main Streamlit script instead, a worker process on
Windows (which uses "spawn") would need to re-import that script to
resolve the reference -- and re-importing it would re-run every
st.title()/st.button()/etc. call at module scope, which is unsafe and can
hang or throw. Keeping this module import-only and Streamlit-free sidesteps
that problem entirely: worker processes import *this* file, never the
Streamlit script.

Nothing in this file should import streamlit or reference st.session_state.
"""

import re

import numpy as np
import pdfplumber
import pymupdf  # PyMuPDF
from scipy import ndimage

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


def _extend_bbox_to_trailing_rows(words, bbox, max_gap=18, max_iter=6):
    """Some Revit-exported schedules have a final data row with no ruling
    lines below it, so pdfplumber's ruled-table detection clips it off.
    Walk downward picking up any word lines that are clearly a continuation
    (same x-range, small vertical gap) of the table. Takes the page's
    already-extracted word list (see detect_regions_on_page) instead of
    re-running extract_words() itself -- that call is the single slowest
    step in detection and this function used to trigger it a second time
    on every single schedule found on a page."""
    x0, y0, x1, y1 = bbox
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


def _detect_schedules(page_pl, page_area, words):
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
        extended_bbox = _extend_bbox_to_trailing_rows(words, (x0, y0, x1, y1))
        schedules.append({"bbox": extended_bbox, "title": title or "Schedule"})

    return schedules


def _detect_notes(words, schedules, page_h):
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


def _detect_view_captions(words, schedules, notes_bbox):
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

    # extract_words() re-derives word groupings from every char on the page
    # and is by far the most expensive single call in detection on a
    # vector/text-dense Revit sheet -- do it exactly once per page and hand
    # the same list to every detector below instead of each one re-parsing
    # the page's text independently (this used to happen 2-3x per page).
    words = page_pl.extract_words()

    schedules = _detect_schedules(page_pl, page_area, words)
    notes_bbox = _detect_notes(words, schedules, page_pl.height)
    views = _detect_view_captions(words, schedules, notes_bbox)

    # _find_drawing_blobs rasterizes the whole page and runs connected-
    # component labeling on it -- the single heaviest step per page after
    # word extraction. It's only ever used to size up View regions, so
    # skip it entirely on schedule/notes-only sheets (common on sheets
    # that are all rebar or embed schedules with no drafting views).
    if views:
        exclude = [s["bbox"] for s in schedules]
        if notes_bbox:
            exclude.append(notes_bbox)
        blobs = _find_drawing_blobs(page_mu, page_area, exclude)
    else:
        blobs = []

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


def detect_one_page_worker(pdf_bytes, page_number):
    """Detects all regions on exactly one page. This is the function each
    ProcessPoolExecutor task calls (see detect_all_pages in the main
    script) -- it opens its own pdfplumber/PyMuPDF handles for just this
    one page, so it's fully self-contained and safe to run in a separate
    process with no shared state. page_number is 1-indexed.

    Returns (page_number, regions, error_message_or_None) -- a page that
    fails to scan reports its error here instead of raising, so one bad
    sheet in a 70+ sheet batch doesn't take down the whole run.
    """
    import io
    idx = page_number - 1
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf_pl:
            page_pl = pdf_pl.pages[idx]
            doc_mu = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            try:
                page_mu = doc_mu[idx]
                regions = detect_regions_on_page(page_pl, page_mu, idx)
            finally:
                doc_mu.close()
                page_pl.close()
        return page_number, regions, None
    except Exception as e:
        return page_number, [], f"{type(e).__name__}: {e}"
