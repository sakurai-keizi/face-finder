#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "insightface",
#   "pillow",
#   "numpy",
#   "onnxruntime",
# ]
# ///

import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from insightface.app import FaceAnalysis


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
SIMILARITY_THRESHOLD = 0.35   # cosine similarity (buffalo_l 512-dim embedding)
THUMB_SIZE = 220               # result thumbnail size (px)
RESULT_COLS = 4                # columns in result grid


# ---------------------------------------------------------------------------
# Directory face database (scanned in background thread)
# ---------------------------------------------------------------------------

class DirectoryFaceDB:
    def __init__(self, directory: Path, face_app):
        self.directory = directory
        self.face_app = face_app
        self.records: list[dict] = []   # {path, face, pil_image}
        self.status = "idle"            # idle / scanning / done / error
        self.scanned = 0
        self.total = 0
        self._lock = threading.Lock()

    def start_scan(self, on_progress=None, on_complete=None):
        self.status = "scanning"
        t = threading.Thread(
            target=self._scan, args=(on_progress, on_complete), daemon=True
        )
        t.start()

    def _scan(self, on_progress, on_complete):
        image_files = sorted(
            p for p in self.directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.total = len(image_files)
        print(f"[Scan] {self.total} image(s) found in '{self.directory}'")
        for i, path in enumerate(image_files):
            if on_progress:
                on_progress(i + 1, self.total, path.name)
            try:
                img = Image.open(path).convert("RGB")
                bgr = np.array(img)[:, :, ::-1].copy()
                faces = self.face_app.get(bgr)
                n = 0
                with self._lock:
                    for face in faces:
                        if face.embedding is not None:
                            self.records.append(
                                {"path": path, "face": face, "pil_image": img}
                            )
                            n += 1
                print(f"[Scan] ({i + 1}/{self.total}) {path.name}  -> {n} face(s)")
            except Exception as e:
                print(f"[Scan] ({i + 1}/{self.total}) {path.name}  -> ERROR: {e}")
        self.scanned = self.total
        self.status = "done"
        print(f"[Scan] Done. Total faces indexed: {len(self.records)}")
        if on_complete:
            on_complete(len(self.records))

    def search(self, query_emb: np.ndarray, threshold=SIMILARITY_THRESHOLD):
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        results = []
        with self._lock:
            records = list(self.records)
        for rec in records:
            e = rec["face"].embedding
            e = e / (np.linalg.norm(e) + 1e-8)
            sim = float(np.dot(q, e))
            if sim >= threshold:
                results.append({**rec, "similarity": sim})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Result window
# ---------------------------------------------------------------------------

def make_face_thumb(pil_image: Image.Image, bbox, thumb_size=THUMB_SIZE) -> Image.Image:
    """Crop the image around bbox with padding, draw bbox, resize to thumb_size."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    bw, bh = x2 - x1, y2 - y1
    pad = max(int(max(bw, bh) * 0.4), 20)
    W, H = pil_image.size
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(W, x2 + pad)
    cy2 = min(H, y2 + pad)
    crop = pil_image.crop((cx1, cy1, cx2, cy2)).copy()
    # Draw bbox relative to crop
    draw = ImageDraw.Draw(crop)
    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1
    draw.rectangle([rx1, ry1, rx2, ry2], outline="yellow", width=3)
    # Fit into thumb_size × thumb_size
    crop.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    result = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 30))
    ox = (thumb_size - crop.width) // 2
    oy = (thumb_size - crop.height) // 2
    result.paste(crop, (ox, oy))
    return result


def open_results_window(root, results: list[dict], query_face_idx: int):
    win = tk.Toplevel(root)
    win.title(f"Search results for Face #{query_face_idx + 1}  ({len(results)} match(es))")
    win.configure(bg="#1e1e1e")

    # Scrollable canvas
    outer = tk.Frame(win, bg="#1e1e1e")
    outer.pack(fill=tk.BOTH, expand=True)

    vscroll = tk.Scrollbar(outer, orient=tk.VERTICAL)
    vscroll.pack(side=tk.RIGHT, fill=tk.Y)

    cv = tk.Canvas(outer, bg="#1e1e1e", yscrollcommand=vscroll.set)
    cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vscroll.config(command=cv.yview)

    inner = tk.Frame(cv, bg="#1e1e1e")
    cv_win = cv.create_window((0, 0), window=inner, anchor=tk.NW)

    def on_configure(event):
        cv.configure(scrollregion=cv.bbox("all"))
        cv.itemconfig(cv_win, width=event.width)

    inner.bind("<Configure>", on_configure)
    cv.bind("<Configure>", on_configure)

    # Bind mouse-wheel scroll
    def _on_mousewheel(event):
        cv.yview_scroll(int(-1 * (event.delta / 120)), "units")

    win.bind("<MouseWheel>", _on_mousewheel)
    win.bind("<Button-4>", lambda e: cv.yview_scroll(-1, "units"))
    win.bind("<Button-5>", lambda e: cv.yview_scroll(1, "units"))

    # Keep PhotoImage references alive
    photo_refs = []

    if not results:
        tk.Label(inner, text="No matching faces found.", bg="#1e1e1e", fg="#aaaaaa",
                 font=("Helvetica", 13)).pack(pady=40, padx=40)
        win.geometry("400x200")
        return

    for idx, rec in enumerate(results):
        col = idx % RESULT_COLS
        row = idx // RESULT_COLS

        frame = tk.Frame(inner, bg="#2a2a2a", bd=1, relief=tk.RIDGE)
        frame.grid(row=row, column=col, padx=6, pady=6, sticky="n")

        thumb = make_face_thumb(rec["pil_image"], rec["face"].bbox)
        photo = ImageTk.PhotoImage(thumb)
        photo_refs.append(photo)

        img_label = tk.Label(frame, image=photo, bg="#2a2a2a", cursor="hand2")
        img_label.pack()

        name = Path(rec["path"]).name
        sim_pct = rec["similarity"] * 100
        tk.Label(
            frame,
            text=f"{name}\n{sim_pct:.1f}%",
            bg="#2a2a2a",
            fg="#cccccc",
            font=("Helvetica", 8),
            wraplength=THUMB_SIZE,
            justify=tk.CENTER,
        ).pack(pady=(2, 4))

        # Click thumbnail → open full image in separate window
        def open_full(event, r=rec):
            _open_full_image(root, r["pil_image"], r["face"].bbox, r["path"])

        img_label.bind("<Button-1>", open_full)

    win.inner_photos = photo_refs  # prevent GC

    # Resize window to fit grid (max 1200 wide)
    cols_actual = min(len(results), RESULT_COLS)
    win_w = min(cols_actual * (THUMB_SIZE + 20) + 30, 1200)
    rows_actual = (len(results) + RESULT_COLS - 1) // RESULT_COLS
    win_h = min(rows_actual * (THUMB_SIZE + 60) + 20, 900)
    win.geometry(f"{win_w}x{win_h}")


def _open_full_image(root, pil_image: Image.Image, bbox, path):
    """Open the full source image with bbox drawn, in its own window."""
    win = tk.Toplevel(root)
    win.title(str(path))
    win.configure(bg="#1e1e1e")

    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    orig_w, orig_h = pil_image.size
    scale = min(screen_w * 0.85 / orig_w, screen_h * 0.85 / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    img = pil_image.resize((disp_w, disp_h), Image.LANCZOS).copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [c * scale for c in bbox]
    draw.rectangle([x1, y1, x2, y2], outline="yellow", width=max(2, int(3 * scale)))

    photo = ImageTk.PhotoImage(img)
    label = tk.Label(win, image=photo, bg="#1e1e1e")
    label.pack()
    label.image = photo
    win.geometry(f"{disp_w}x{disp_h}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run face_finder.py <image_file> [search_directory]")
        sys.exit(1)

    image_path = sys.argv[1]
    search_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    if search_dir and not search_dir.is_dir():
        print(f"Error: '{search_dir}' is not a directory")
        sys.exit(1)

    # Load main image
    pil_image = Image.open(image_path).convert("RGB")
    bgr_image = np.array(pil_image)[:, :, ::-1].copy()

    print("Loading InsightFace model...")
    face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("Detecting faces in main image...")
    faces = face_app.get(bgr_image)
    print(f"Found {len(faces)} face(s)")

    # --- GUI setup ---
    root = tk.Tk()
    root.title(f"Face Finder - {image_path}")

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    orig_w, orig_h = pil_image.size
    scale = min(screen_w * 0.7 / orig_w, screen_h * 0.9 / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    display_image = pil_image.resize((disp_w, disp_h), Image.LANCZOS)
    photo = ImageTk.PhotoImage(display_image)

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="crosshair")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

    box_items = []
    for face in faces:
        x1, y1, x2, y2 = [c * scale for c in face.bbox]
        item = canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=2)
        box_items.append(item)

    # Right panel
    info_frame = tk.Frame(root, width=320, bg="#1e1e1e")
    info_frame.pack(side=tk.RIGHT, fill=tk.Y)
    info_frame.pack_propagate(False)

    header_var = tk.StringVar(value=f"{len(faces)} face(s) detected\nClick a face to inspect")
    tk.Label(
        info_frame, textvariable=header_var,
        bg="#1e1e1e", fg="#cccccc", font=("Helvetica", 11), justify=tk.CENTER,
    ).pack(pady=(12, 6))

    info_text = scrolledtext.ScrolledText(
        info_frame, wrap=tk.WORD, font=("Courier", 9),
        bg="#2d2d2d", fg="#d4d4d4", insertbackground="white", state=tk.DISABLED,
    )
    info_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    status_var = tk.StringVar(value="Ready")
    tk.Label(info_frame, textvariable=status_var, bg="#1e1e1e", fg="#888888",
             font=("Helvetica", 9)).pack(pady=4)

    # --- Directory face DB (optional) ---
    face_db: DirectoryFaceDB | None = None

    if search_dir:
        face_db = DirectoryFaceDB(search_dir, face_app)
        status_var.set(f"Scanning {search_dir.name} ...")

        def on_progress(done, total, name):
            root.after(0, status_var.set, f"Scanning {done}/{total}: {name}")

        def on_complete(n_faces):
            root.after(0, status_var.set, f"Ready  ({n_faces} faces indexed)")
            root.after(0, header_var.set,
                       f"{len(faces)} face(s) detected\nClick to inspect / search")

        face_db.start_scan(on_progress=on_progress, on_complete=on_complete)

    # --- Helpers ---
    def set_info(text):
        info_text.config(state=tk.NORMAL)
        info_text.delete("1.0", tk.END)
        info_text.insert(tk.END, text)
        info_text.config(state=tk.DISABLED)

    def highlight_face(idx):
        for i, item in enumerate(box_items):
            canvas.itemconfig(item,
                              outline="yellow" if i == idx else "lime",
                              width=3 if i == idx else 2)

    def on_click(event):
        orig_x = event.x / scale
        orig_y = event.y / scale

        clicked_idx = None
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                clicked_idx = i
                break

        if clicked_idx is None:
            status_var.set("No face at click location")
            set_info("No face detected at that location.\n\nClick inside a highlighted bounding box.")
            for item in box_items:
                canvas.itemconfig(item, outline="lime", width=2)
            return

        highlight_face(clicked_idx)
        face = faces[clicked_idx]
        status_var.set(f"Face #{clicked_idx + 1} selected")

        # Build info text
        lines = [f"=== Face #{clicked_idx + 1} ===\n"]
        bbox = [int(c) for c in face.bbox]
        lines.append(f"Bounding box:\n  x1={bbox[0]}, y1={bbox[1]}\n  x2={bbox[2]}, y2={bbox[3]}\n")
        if face.det_score is not None:
            lines.append(f"Detection score: {face.det_score:.4f}\n")
        if face.gender is not None:
            lines.append(f"Gender:          {'Male' if face.gender == 1 else 'Female'}\n")
        if face.age is not None:
            lines.append(f"Age (estimated): {face.age}\n")
        if face.embedding is not None:
            emb = face.embedding
            lines.append(f"\nEmbedding dim: {len(emb)}  norm: {np.linalg.norm(emb):.4f}\n")
        set_info("".join(lines))

        # Search directory if available
        if face_db and face.embedding is not None:
            if face_db.status != "done":
                status_var.set("Still scanning — please wait...")
                return
            status_var.set("Searching...")

            def do_search():
                results = face_db.search(face.embedding)
                root.after(0, _show_results, results, clicked_idx)

            threading.Thread(target=do_search, daemon=True).start()

    def _show_results(results, face_idx):
        status_var.set(f"{len(results)} match(es) found")
        open_results_window(root, results, face_idx)

    canvas.bind("<Button-1>", on_click)
    root.mainloop()


if __name__ == "__main__":
    main()
