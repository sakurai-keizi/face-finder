#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "insightface",
#   "pillow",
#   "numpy",
#   "onnxruntime-gpu",
# ]
# ///

import sys
import time
import hashlib
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext, ttk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from insightface.app import FaceAnalysis
import onnxruntime


IMAGE_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
SIMILARITY_THRESHOLD = 0.35   # 同一人物と判定するコサイン類似度の閾値
PLAUSIBLE_THRESHOLD  = 0.20   # 元画像で点線BBOXを付ける下限閾値
THUMB_SIZE           = 220
RESULT_COLS          = 4
CACHE_FILENAME       = ".face_finder_cache.pkl"
CACHE_VERSION        = 1


# ---------------------------------------------------------------------------
# キャッシュ用データクラス（InsightFace face オブジェクトの代替）
# ---------------------------------------------------------------------------

@dataclass
class CachedFace:
    bbox:      np.ndarray
    embedding: np.ndarray
    det_score: float             = 0.0
    gender:    int | None        = None
    age:       int | None        = None
    kps:       np.ndarray | None = None


# ---------------------------------------------------------------------------
# キャッシュ I/O
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict[str, list[CachedFace]]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and data.get("__version__") == CACHE_VERSION:
            return data.get("faces", {})
    except Exception as e:
        print(f"[Cache] Load error ({e}), starting fresh")
    return {}


def _save_cache(cache_path: Path, cache: dict[str, list[CachedFace]]) -> None:
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"__version__": CACHE_VERSION, "faces": cache}, f)
    except Exception as e:
        print(f"[Cache] Save error: {e}")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Directory face database
# ---------------------------------------------------------------------------

class DirectoryFaceDB:
    def __init__(self, directory: Path, face_app, face_app_lock: threading.Lock):
        self.directory     = directory
        self.face_app      = face_app
        self.face_app_lock = face_app_lock
        self.cache_path    = directory / CACHE_FILENAME
        self._cache        = _load_cache(self.cache_path)
        self._cache_dirty  = False
        self.records: list[dict] = []   # {path, face (CachedFace), pil_image}
        self.status  = "idle"
        self.scanned = 0
        self.total   = 0
        self._lock   = threading.Lock()

        n = sum(len(v) for v in self._cache.values())
        print(f"[Cache] Loaded {len(self._cache)} image(s) / {n} face(s) from cache")

    def start_scan(self, on_progress=None, on_complete=None):
        self.status = "scanning"
        threading.Thread(
            target=self._scan, args=(on_progress, on_complete), daemon=True
        ).start()

    def _scan(self, on_progress, on_complete):
        image_files = sorted(
            p for p in self.directory.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.total = len(image_files)
        print(f"[Scan] {self.total} image(s) found in '{self.directory}'")

        for i, path in enumerate(image_files):
            if on_progress:
                on_progress(i + 1, self.total, path.name)
            try:
                img_hash = _hash_file(path)

                if img_hash in self._cache:
                    # ---- キャッシュヒット ----
                    cached = self._cache[img_hash]
                    img = Image.open(path).convert("RGB")
                    with self._lock:
                        for cf in cached:
                            self.records.append(
                                {"path": path, "face": cf, "pil_image": img}
                            )
                    print(f"[Scan] ({i + 1}/{self.total}) {path.name}"
                          f"  -> {len(cached)} face(s)  [cache]")
                else:
                    # ---- 新規: InsightFace で検出 ----
                    img = Image.open(path).convert("RGB")
                    bgr = np.array(img)[:, :, ::-1].copy()
                    with self.face_app_lock:
                        raw_faces = self.face_app.get(bgr)

                    new_faces: list[CachedFace] = []
                    for f in raw_faces:
                        if f.embedding is not None:
                            new_faces.append(CachedFace(
                                bbox      = f.bbox.copy(),
                                embedding = f.embedding.copy(),
                                det_score = float(f.det_score) if f.det_score is not None else 0.0,
                                gender    = int(f.gender) if f.gender is not None else None,
                                age       = int(f.age)    if f.age    is not None else None,
                                kps       = f.kps.copy()  if f.kps    is not None else None,
                            ))

                    self._cache[img_hash] = new_faces
                    self._cache_dirty = True

                    with self._lock:
                        for cf in new_faces:
                            self.records.append(
                                {"path": path, "face": cf, "pil_image": img}
                            )
                    print(f"[Scan] ({i + 1}/{self.total}) {path.name}"
                          f"  -> {len(new_faces)} face(s)")

            except Exception as e:
                print(f"[Scan] ({i + 1}/{self.total}) {path.name}  -> ERROR: {e}")

        if self._cache_dirty:
            _save_cache(self.cache_path, self._cache)
            print(f"[Cache] Saved to {self.cache_path}")

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
        with self._lock:
            return [r for r in self.records if r["path"] == path]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_dashed_rect(draw: ImageDraw.ImageDraw, bbox, color, width=2, dash=10):
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

    dashed_line(x1, y1, x2, y1)
    dashed_line(x2, y1, x2, y2)
    dashed_line(x2, y2, x1, y2)
    dashed_line(x1, y2, x1, y1)


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
    win = tk.Toplevel(root)
    win.title(str(path))
    win.configure(bg="#1e1e1e")

    orig_w, orig_h = pil_image.size
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()

    # 初期ウィンドウサイズ: 原寸。ただし画面をはみ出す場合は縮小
    init_scale = min(screen_w * 0.9 / orig_w, screen_h * 0.9 / orig_h, 1.0)
    init_w = int(orig_w * init_scale)
    init_h = int(orig_h * init_scale)
    win.geometry(f"{init_w}x{init_h}")
    win.state("normal")  # 最大化を防ぐ

    canvas = tk.Canvas(win, bg="#1e1e1e", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    # BBOXを描画した画像を生成（スケール引数に応じて）
    def _render(scale: float) -> Image.Image:
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        img  = pil_image.resize((disp_w, disp_h), Image.LANCZOS).copy()
        draw = ImageDraw.Draw(img)

        # 点線BBOX: 有り得そうな他の顔
        if face_db is not None and query_emb is not None:
            q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            for rec in face_db.faces_in_image(path):
                e   = rec["face"].embedding
                e   = e / (np.linalg.norm(e) + 1e-8)
                sim = float(np.dot(q, e))
                rx1, ry1 = rec["face"].bbox[0], rec["face"].bbox[1]
                mx1, my1 = matched_bbox[0], matched_bbox[1]
                if not (abs(rx1 - mx1) < 5 and abs(ry1 - my1) < 5) and sim >= PLAUSIBLE_THRESHOLD:
                    sx1, sy1, sx2, sy2 = [c * scale for c in rec["face"].bbox]
                    draw_dashed_rect(draw, [sx1, sy1, sx2, sy2],
                                     color="orange", width=max(2, int(2 * scale)), dash=10)

        # 実線BBOX: マッチした顔
        x1, y1, x2, y2 = [c * scale for c in matched_bbox]
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=max(2, int(3 * scale)))
        return img

    photo_ref   = [None]
    resize_after = [None]

    def _redraw(event=None):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        scale = min(cw / orig_w, ch / orig_h)
        img   = _render(scale)
        photo = ImageTk.PhotoImage(img)
        photo_ref[0] = photo
        canvas.delete("all")
        ox = (cw - img.width)  // 2
        oy = (ch - img.height) // 2
        canvas.create_image(ox, oy, anchor=tk.NW, image=photo)

    def _on_resize(event):
        # リサイズ中の連続イベントをまとめて 80ms 後に一度だけ再描画
        if resize_after[0]:
            win.after_cancel(resize_after[0])
        resize_after[0] = win.after(80, _redraw)

    canvas.bind("<Configure>", _on_resize)


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

    trt_cache_dir = Path.home() / ".cache" / "face_finder" / "trt"
    trt_cache_dir.mkdir(parents=True, exist_ok=True)

    trt_available  = "TensorrtExecutionProvider" in onnxruntime.get_available_providers()
    trt_needs_build = trt_available and not any(trt_cache_dir.glob("*.engine"))

    # --- ローディング画面（GUI を先に起動）---
    root = tk.Tk()
    root.title("Face Finder - 初期化中")
    root.configure(bg="#1e1e1e")
    root.resizable(False, False)

    load_frame = tk.Frame(root, bg="#1e1e1e", padx=48, pady=36)
    load_frame.pack()

    if trt_needs_build:
        title_text  = "TensorRT エンジンを初回ビルド中"
        detail_text = "初回起動時のみ数分かかります。\n次回以降はキャッシュが使われ数秒で起動します。"
        print("[Init] TensorRT engine build required (first run). This may take several minutes.")
    elif trt_available:
        title_text  = "TensorRT エンジンを読み込み中"
        detail_text = "しばらくお待ちください。"
        print("[Init] Loading TensorRT engine from cache.")
    else:
        title_text  = "InsightFace モデルを読み込み中"
        detail_text = "しばらくお待ちください。"
        print("[Init] Loading InsightFace model.")

    tk.Label(load_frame, text=title_text, bg="#1e1e1e", fg="white",
             font=("Helvetica", 14, "bold")).pack(pady=(0, 8))
    tk.Label(load_frame, text=detail_text, bg="#1e1e1e", fg="#aaaaaa",
             font=("Helvetica", 10), justify=tk.CENTER).pack(pady=(0, 20))

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure("dark.Horizontal.TProgressbar",
                    troughcolor="#2d2d2d", background="#4a9eff", borderwidth=0)
    progress = ttk.Progressbar(load_frame, mode="indeterminate", length=320,
                               style="dark.Horizontal.TProgressbar")
    progress.pack(pady=(0, 14))
    progress.start(40)

    elapsed_var = tk.StringVar(value="経過: 0 秒")
    tk.Label(load_frame, textvariable=elapsed_var, bg="#1e1e1e", fg="#666666",
             font=("Helvetica", 9)).pack()

    root.geometry("420x220")
    root.update()

    start_time = time.time()

    def _tick():
        elapsed = int(time.time() - start_time)
        elapsed_var.set(f"経過: {elapsed} 秒")
        root.after(1000, _tick)

    root.after(1000, _tick)

    # --- バックグラウンドで InsightFace 初期化 ---
    init_result: dict = {}

    def _do_init():
        face_app = FaceAnalysis(providers=[
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path":   str(trt_cache_dir),
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        init_result["face_app"] = face_app
        root.after(0, _on_init_done)

    threading.Thread(target=_do_init, daemon=True).start()

    def _on_init_done():
        elapsed = time.time() - start_time
        print(f"[Init] Done in {elapsed:.1f}s")
        progress.stop()
        load_frame.destroy()
        _setup_main(root, image_path, search_dir, init_result["face_app"])

    root.mainloop()


def _setup_main(root: tk.Tk, image_path: str, search_dir: Path | None, face_app):
    face_app_lock = threading.Lock()

    # スキャンを即時開始
    face_db: DirectoryFaceDB | None = None
    if search_dir:
        face_db = DirectoryFaceDB(search_dir, face_app, face_app_lock)
        face_db.start_scan()

    # メイン画像の顔検出
    print("Detecting faces in main image...")
    pil_image = Image.open(image_path).convert("RGB")
    bgr_image = np.array(pil_image)[:, :, ::-1].copy()
    with face_app_lock:
        faces = face_app.get(bgr_image)
    print(f"Found {len(faces)} face(s)")

    root.title(f"Face Finder - {image_path}")
    root.resizable(True, True)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    orig_w, orig_h = pil_image.size
    INFO_PANEL_W = 320
    scale  = min((screen_w - INFO_PANEL_W) * 0.95 / orig_w, screen_h * 0.95 / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    root.geometry(f"{disp_w + INFO_PANEL_W}x{disp_h}")

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

    if face_db:
        header_var.set(f"{len(faces)} face(s) detected\nClick to inspect / search")
        if face_db.status == "done":
            status_var.set(f"Ready  ({len(face_db.records)} faces indexed)")
        else:
            status_var.set(f"Scanning {search_dir.name} ...")

            def _poll_scan():
                if face_db.status == "done":
                    status_var.set(f"Ready  ({len(face_db.records)} faces indexed)")
                else:
                    if face_db.total > 0:
                        status_var.set(f"Scanning {face_db.scanned}/{face_db.total} ...")
                    root.after(300, _poll_scan)

            root.after(300, _poll_scan)

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


if __name__ == "__main__":
    main()
