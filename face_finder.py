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
SIMILARITY_THRESHOLD = 0.35   # 同一人物と判定するコサイン類似度の閾値
PLAUSIBLE_THRESHOLD  = 0.20   # 元画像で点線BBOXを付ける下限閾値
THUMB_SIZE  = 220
RESULT_COLS = 4


# ---------------------------------------------------------------------------
# Directory face database
# ---------------------------------------------------------------------------

class DirectoryFaceDB:
    def __init__(self, directory: Path, face_app):
        self.directory = directory
        self.face_app  = face_app
        self.records: list[dict] = []   # {path, face, pil_image}
        self.status = "idle"
        self.scanned = 0
        self.total   = 0
        self._lock   = threading.Lock()

    def start_scan(self, on_progress=None, on_complete=None):
        self.status = "scanning"
        threading.Thread(
            target=self._scan, args=(on_progress, on_complete), daemon=True
        ).start()

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
                img  = Image.open(path).convert("RGB")
                bgr  = np.array(img)[:, :, ::-1].copy()
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
        self.status  = "done"
        print(f"[Scan] Done. Total faces indexed: {len(self.records)}")
        if on_complete:
            on_complete(len(self.records))

    def search(self, query_emb: np.ndarray, threshold=SIMILARITY_THRESHOLD) -> list[dict]:
        """画像ごとに最も類似度の高い顔を1件だけ返す（降順）。"""
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        best_per_image: dict[Path, dict] = {}
        with self._lock:
            records = list(self.records)
        for rec in records:
            e   = rec["face"].embedding
            e   = e / (np.linalg.norm(e) + 1e-8)
            sim = float(np.dot(q, e))
            if sim < threshold:
                continue
            key = rec["path"]
            if key not in best_per_image or sim > best_per_image[key]["similarity"]:
                best_per_image[key] = {**rec, "similarity": sim}
        return sorted(best_per_image.values(), key=lambda x: x["similarity"], reverse=True)

    def faces_in_image(self, path: Path) -> list:
        """指定画像に含まれる全顔レコードを返す。"""
        with self._lock:
            return [r for r in self.records if r["path"] == path]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_dashed_rect(draw: ImageDraw.ImageDraw, bbox, color, width=2, dash=10):
    """PIL には点線が無いので手動で描画する。"""
    x1, y1, x2, y2 = [int(c) for c in bbox]

    def dashed_line(x0, y0, x1e, y1e):
        length = ((x1e - x0) ** 2 + (y1e - y0) ** 2) ** 0.5
        if length == 0:
            return
        dx, dy = (x1e - x0) / length, (y1e - y0) / length
        i, drawing = 0.0, True
        while i < length:
            end = min(i + dash, length)
            if drawing:
                draw.line(
                    [(x0 + dx * i, y0 + dy * i), (x0 + dx * end, y0 + dy * end)],
                    fill=color, width=width,
                )
            i += dash
            drawing = not drawing

    dashed_line(x1, y1, x2, y1)  # 上
    dashed_line(x2, y1, x2, y2)  # 右
    dashed_line(x2, y2, x1, y2)  # 下
    dashed_line(x1, y2, x1, y1)  # 左


# ---------------------------------------------------------------------------
# Result thumbnail
# ---------------------------------------------------------------------------

def make_face_thumb(pil_image: Image.Image, bbox, thumb_size=THUMB_SIZE) -> Image.Image:
    x1, y1, x2, y2 = [int(c) for c in bbox]
    bw, bh = x2 - x1, y2 - y1
    pad    = max(int(max(bw, bh) * 0.4), 20)
    W, H   = pil_image.size
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
    crop = pil_image.crop((cx1, cy1, cx2, cy2)).copy()
    draw = ImageDraw.Draw(crop)
    draw.rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline="yellow", width=3)
    crop.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    result = Image.new("RGB", (thumb_size, thumb_size), (30, 30, 30))
    result.paste(crop, ((thumb_size - crop.width) // 2, (thumb_size - crop.height) // 2))
    return result


# ---------------------------------------------------------------------------
# Full image window
# ---------------------------------------------------------------------------

def _open_full_image(root, pil_image: Image.Image, matched_bbox, path,
                     face_db: "DirectoryFaceDB | None" = None,
                     query_emb: "np.ndarray | None" = None):
    """
    元画像を開く。
    - matched_bbox の顔: 実線・黄色
    - 同じ画像内の他の顔で PLAUSIBLE_THRESHOLD 以上のもの: 点線・オレンジ
    """
    win = tk.Toplevel(root)
    win.title(str(path))
    win.configure(bg="#1e1e1e")

    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    orig_w, orig_h = pil_image.size
    scale  = min(screen_w * 0.85 / orig_w, screen_h * 0.85 / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    img  = pil_image.resize((disp_w, disp_h), Image.LANCZOS).copy()
    draw = ImageDraw.Draw(img)

    # 点線BBOX: 同画像内の他の顔で有り得そうなもの
    if face_db is not None and query_emb is not None:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        for rec in face_db.faces_in_image(path):
            e   = rec["face"].embedding
            e   = e / (np.linalg.norm(e) + 1e-8)
            sim = float(np.dot(q, e))
            # matched_bbox と重複する顔はスキップ（実線で描くため）
            rx1, ry1, rx2, ry2 = rec["face"].bbox
            mx1, my1, mx2, my2 = matched_bbox
            is_matched = (abs(rx1 - mx1) < 5 and abs(ry1 - my1) < 5)
            if not is_matched and sim >= PLAUSIBLE_THRESHOLD:
                sx1, sy1, sx2, sy2 = [c * scale for c in rec["face"].bbox]
                draw_dashed_rect(draw, [sx1, sy1, sx2, sy2],
                                 color="orange", width=max(2, int(2 * scale)), dash=10)

    # 実線BBOX: マッチした顔
    lw = max(2, int(3 * scale))
    x1, y1, x2, y2 = [c * scale for c in matched_bbox]
    draw.rectangle([x1, y1, x2, y2], outline="yellow", width=lw)

    photo = ImageTk.PhotoImage(img)
    label = tk.Label(win, image=photo, bg="#1e1e1e")
    label.pack()
    label.image = photo
    win.geometry(f"{disp_w}x{disp_h}")


# ---------------------------------------------------------------------------
# Results window
# ---------------------------------------------------------------------------

def open_results_window(root, results: list[dict], query_face_idx: int,
                        face_db=None, query_emb=None):
    win = tk.Toplevel(root)
    win.title(f"Search results for Face #{query_face_idx + 1}  ({len(results)} match(es))")
    win.configure(bg="#1e1e1e")

    outer   = tk.Frame(win, bg="#1e1e1e")
    outer.pack(fill=tk.BOTH, expand=True)
    vscroll = tk.Scrollbar(outer, orient=tk.VERTICAL)
    vscroll.pack(side=tk.RIGHT, fill=tk.Y)
    cv = tk.Canvas(outer, bg="#1e1e1e", yscrollcommand=vscroll.set)
    cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vscroll.config(command=cv.yview)

    inner  = tk.Frame(cv, bg="#1e1e1e")
    cv_win = cv.create_window((0, 0), window=inner, anchor=tk.NW)

    def on_configure(event):
        cv.configure(scrollregion=cv.bbox("all"))
        cv.itemconfig(cv_win, width=event.width)

    inner.bind("<Configure>", on_configure)
    cv.bind("<Configure>", on_configure)
    win.bind("<MouseWheel>", lambda e: cv.yview_scroll(int(-1 * e.delta / 120), "units"))
    win.bind("<Button-4>",   lambda e: cv.yview_scroll(-1, "units"))
    win.bind("<Button-5>",   lambda e: cv.yview_scroll(1,  "units"))

    photo_refs = []

    if not results:
        tk.Label(inner, text="No matching faces found.", bg="#1e1e1e", fg="#aaaaaa",
                 font=("Helvetica", 13)).pack(pady=40, padx=40)
        win.geometry("400x200")
        return

    for idx, rec in enumerate(results):
        frame = tk.Frame(inner, bg="#2a2a2a", bd=1, relief=tk.RIDGE)
        frame.grid(row=idx // RESULT_COLS, column=idx % RESULT_COLS, padx=6, pady=6, sticky="n")

        thumb = make_face_thumb(rec["pil_image"], rec["face"].bbox)
        photo = ImageTk.PhotoImage(thumb)
        photo_refs.append(photo)

        img_label = tk.Label(frame, image=photo, bg="#2a2a2a", cursor="hand2")
        img_label.pack()

        sim_pct = rec["similarity"] * 100
        tk.Label(
            frame,
            text=f"{Path(rec['path']).name}\n{sim_pct:.1f}%",
            bg="#2a2a2a", fg="#cccccc", font=("Helvetica", 8),
            wraplength=THUMB_SIZE, justify=tk.CENTER,
        ).pack(pady=(2, 4))

        def open_full(event, r=rec):
            _open_full_image(root, r["pil_image"], r["face"].bbox, r["path"],
                             face_db=face_db, query_emb=query_emb)

        img_label.bind("<Button-1>", open_full)

    win.inner_photos = photo_refs

    cols_actual = min(len(results), RESULT_COLS)
    win_w = min(cols_actual * (THUMB_SIZE + 20) + 30, 1200)
    rows_actual = (len(results) + RESULT_COLS - 1) // RESULT_COLS
    win_h = min(rows_actual * (THUMB_SIZE + 60) + 20, 900)
    win.geometry(f"{win_w}x{win_h}")


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

    pil_image = Image.open(image_path).convert("RGB")
    bgr_image = np.array(pil_image)[:, :, ::-1].copy()

    print("Loading InsightFace model...")
    face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("Detecting faces in main image...")
    faces = face_app.get(bgr_image)
    print(f"Found {len(faces)} face(s)")

    # --- GUI ---
    root = tk.Tk()
    root.title(f"Face Finder - {image_path}")

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    orig_w, orig_h = pil_image.size
    scale  = min(screen_w * 0.7 / orig_w, screen_h * 0.9 / orig_h, 1.0)
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
        box_items.append(canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=2))

    # Right panel
    info_frame = tk.Frame(root, width=320, bg="#1e1e1e")
    info_frame.pack(side=tk.RIGHT, fill=tk.Y)
    info_frame.pack_propagate(False)

    header_var = tk.StringVar(value=f"{len(faces)} face(s) detected\nClick a face to inspect")
    tk.Label(info_frame, textvariable=header_var, bg="#1e1e1e", fg="#cccccc",
             font=("Helvetica", 11), justify=tk.CENTER).pack(pady=(12, 6))

    info_text = scrolledtext.ScrolledText(
        info_frame, wrap=tk.WORD, font=("Courier", 9),
        bg="#2d2d2d", fg="#d4d4d4", insertbackground="white", state=tk.DISABLED,
    )
    info_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    status_var = tk.StringVar(value="Ready")
    tk.Label(info_frame, textvariable=status_var, bg="#1e1e1e", fg="#888888",
             font=("Helvetica", 9)).pack(pady=4)

    # Directory face DB
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

    # Helpers
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

        if face_db and face.embedding is not None:
            if face_db.status != "done":
                status_var.set("Still scanning — please wait...")
                return
            status_var.set("Searching...")
            query_emb = face.embedding.copy()

            def do_search():
                results = face_db.search(query_emb)
                root.after(0, _show_results, results, clicked_idx, query_emb)

            threading.Thread(target=do_search, daemon=True).start()

    def _show_results(results, face_idx, query_emb):
        status_var.set(f"{len(results)} match(es) found")
        open_results_window(root, results, face_idx, face_db=face_db, query_emb=query_emb)

    canvas.bind("<Button-1>", on_click)
    root.mainloop()


if __name__ == "__main__":
    main()
