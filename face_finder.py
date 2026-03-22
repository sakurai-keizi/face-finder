#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "insightface",
#   "pillow",
#   "numpy",
#   "onnxruntime-gpu",
#   "tkinterdnd2",
#   "sam2",
#   "huggingface_hub",
#   "ultralytics",
#   "scipy",
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
from tkinterdnd2 import TkinterDnD, DND_FILES

# tkinterdnd2 の既知バグ: DnD完了後にソースウィンドウへ XdndFinished を
# 送り返す際、そのウィンドウが既に破棄されていると BadWindow エラーが出る。
# X11 エラーハンドラを上書きして無視する。
try:
    import ctypes
    _xlib = ctypes.CDLL("libX11.so.6")
    _X_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    _xlib.XSetErrorHandler(_X_ERROR_HANDLER(lambda d, e: 0))
except Exception:
    pass


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
        self.cache_path    = Path.cwd() / CACHE_FILENAME
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

    def search(self, query_emb: np.ndarray, threshold=SIMILARITY_THRESHOLD,
               exclude_path: Path | None = None) -> list[dict]:
        """画像ごとに最も類似度の高い顔を1件だけ返す（降順）。
        exclude_path が指定された場合、そのファイルは結果から除外する。"""
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        exclude = exclude_path.resolve() if exclude_path else None
        best_per_image: dict[Path, dict] = {}
        with self._lock:
            records = list(self.records)
        for rec in records:
            if exclude and rec["path"].resolve() == exclude:
                continue
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
# Similarity color helper
# ---------------------------------------------------------------------------

def _sim_color(sim: float) -> str:
    """類似度を4段階の色に変換する。
    70%以上: 青 / 60-70%: 緑 / 45-60%: 黄 / 45%未満: 赤"""
    pct = sim * 100
    if pct >= 70:
        return "#4488ff"  # 青
    elif pct >= 60:
        return "#44cc44"  # 緑
    elif pct >= 45:
        return "#ffcc00"  # 黄
    else:
        return "#ff4444"  # 赤


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
                     query_emb: "np.ndarray | None" = None,
                     sam2_predictor=None, sam2_lock: "threading.Lock | None" = None,
                     yolo_model=None, sam2_extra: "dict | None" = None):
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
    # GNOME/X11 では geometry を after() で遅延設定しないと最大化が上書きされる
    def _set_geometry():
        win.state("normal")
        win.attributes("-zoomed", False)
        win.geometry(f"{init_w}x{init_h}")
    win.after(100, _set_geometry)

    toolbar = tk.Frame(win, bg="#2a2a2a", height=32)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar.pack_propagate(False)

    canvas = tk.Canvas(win, bg="#1e1e1e", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    # SAM2 セグメンテーションマスク（バックグラウンドで取得）
    seg_mask: list[np.ndarray | None] = [None]
    current_mode = ["large"]  # "small" | "large"

    def _run_sam2():
        import torch, traceback
        # モードに応じてモデルを選択（large は遅延ロード）
        if current_mode[0] == "large":
            lock = sam2_lock if sam2_lock is not None else threading.Lock()
            predictor = (sam2_extra or {}).get("large")
            if predictor is None:
                try:
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    print("[SAM2] Loading hiera-large (first use, please wait)...")
                    with lock:
                        predictor = SAM2ImagePredictor.from_pretrained(
                            "facebook/sam2.1-hiera-large", device="cpu"
                        )
                    if sam2_extra is not None:
                        sam2_extra["large"] = predictor
                    print("[SAM2] hiera-large loaded")
                except Exception as e:
                    print(f"[SAM2] Large model load failed: {e}")
                    traceback.print_exc()
                    return
        else:
            predictor = sam2_predictor
        if predictor is None:
            return
        x1, y1, x2, y2 = [int(c) for c in matched_bbox]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        img_arr = np.array(pil_image)
        try:
            lock = sam2_lock if sam2_lock is not None else threading.Lock()

            # --- YOLOv8 Pose でキーポイントと人物 BBOX を取得 ---
            point_coords = [[cx, cy]]
            point_labels = [1]
            fw, fh = x2 - x1, y2 - y1
            # YOLO 失敗時のフォールバック（全体画像・推定ボックス）
            sam2_img  = img_arr
            sam2_box  = [max(0, x1 - fw), max(0, y1 - int(fh * 0.5)),
                         min(pil_image.width, x2 + fw),
                         min(pil_image.height, y2 + int(fh * 6))]
            crop_offset = (0, 0)

            if yolo_model is not None:
                with lock:
                    yolo_res = yolo_model(img_arr, verbose=False)
                if yolo_res and yolo_res[0].keypoints is not None:
                    boxes    = yolo_res[0].boxes.xyxy.cpu().numpy()
                    kps_xy   = yolo_res[0].keypoints.xy.cpu().numpy()
                    kps_conf = yolo_res[0].keypoints.conf
                    if kps_conf is not None:
                        kps_conf = kps_conf.cpu().numpy()
                    # 顔 BBOX と最もオーバーラップする人物を選ぶ
                    best_idx, best_overlap = -1, 0.0
                    for i, (bx1_, by1_, bx2_, by2_) in enumerate(boxes):
                        ix = max(0, min(x2, bx2_) - max(x1, bx1_))
                        iy = max(0, min(y2, by2_) - max(y1, by1_))
                        overlap = ix * iy / max((x2-x1)*(y2-y1), 1)
                        if overlap > best_overlap:
                            best_overlap, best_idx = overlap, i
                    if best_idx >= 0 and best_overlap > 0.3:
                        bx1_, by1_, bx2_, by2_ = boxes[best_idx]
                        # キーポイントを前景点に
                        kps  = kps_xy[best_idx]
                        conf = kps_conf[best_idx] if kps_conf is not None \
                               else np.ones(len(kps))
                        valid = [(float(kp[0]), float(kp[1]))
                                 for kp, c in zip(kps, conf)
                                 if c > 0.3 and kp[0] > 0 and kp[1] > 0]
                        if valid:
                            point_coords = valid
                            point_labels = [1] * len(valid)
                        # 人物 BBOX を 1.2倍拡張して crop
                        mx = (bx2_ - bx1_) * 0.1
                        my = (by2_ - by1_) * 0.1
                        ox = max(0, int(bx1_ - mx))
                        oy = max(0, int(by1_ - my))
                        ox2 = min(pil_image.width,  int(bx2_ + mx))
                        oy2 = min(pil_image.height, int(by2_ + my))
                        sam2_img     = img_arr[oy:oy2, ox:ox2]
                        crop_offset  = (ox, oy)
                        # 座標を crop 基準に変換
                        point_coords = [(kpx - ox, kpy - oy)
                                        for kpx, kpy in point_coords]
                        sam2_box     = [bx1_ - ox, by1_ - oy,
                                        bx2_ - ox, by2_ - oy]
                        print(f"[YOLO] {len(valid)} kps, crop={sam2_img.shape[:2]}")

            # --- SAM2 推論 ---
            with lock:
                with torch.inference_mode():
                    predictor.set_image(sam2_img)
                    masks, scores, _ = predictor.predict(
                        point_coords=np.array(point_coords),
                        point_labels=np.array(point_labels),
                        box=np.array(sam2_box),
                        multimask_output=True,
                    )
            from scipy.ndimage import binary_fill_holes
            crop_mask = binary_fill_holes(masks[scores.argmax()])

            # crop マスクを元画像サイズに展開
            if crop_offset == (0, 0) and crop_mask.shape == (pil_image.height, pil_image.width):
                seg_mask[0] = crop_mask
            else:
                full_mask = np.zeros((pil_image.height, pil_image.width), dtype=bool)
                ox, oy = crop_offset
                ch, cw = crop_mask.shape
                full_mask[oy:oy+ch, ox:ox+cw] = crop_mask
                seg_mask[0] = full_mask
            print(f"[SAM2] Segmentation done for {path.name}")
            win.after(0, _redraw)
        except Exception as e:
            print(f"[SAM2] Error: {e}")
            traceback.print_exc()

    # BBOXを描画した画像を生成（スケール引数に応じて）
    def _render(scale: float) -> Image.Image:
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        img  = pil_image.resize((disp_w, disp_h), Image.LANCZOS).copy()

        # SAM2 マスクオーバーレイ（シアン半透明）
        if seg_mask[0] is not None:
            mask_img = Image.fromarray(
                (seg_mask[0].astype(np.uint8) * 255)
            ).resize((disp_w, disp_h), Image.NEAREST)
            mask_arr = np.array(mask_img) > 128
            img_arr  = np.array(img).astype(np.float32)
            yellow   = np.array([255, 220, 0], dtype=np.float32)
            img_arr[mask_arr] = img_arr[mask_arr] * 0.55 + yellow * 0.45
            img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

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

        # 実線BBOX: マスク未取得中のみ表示
        if seg_mask[0] is None:
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

    mode_label = tk.StringVar(value="標準モードへ切替")

    def _toggle_mode():
        new_mode = "large" if current_mode[0] == "small" else "small"
        current_mode[0] = new_mode
        mode_label.set("標準モードへ切替" if new_mode == "large" else "高精度モードへ切替")
        seg_mask[0] = None
        _redraw()
        threading.Thread(target=_run_sam2, daemon=True).start()

    tk.Button(
        toolbar, textvariable=mode_label,
        bg="#444", fg="white", relief=tk.FLAT, padx=10,
        command=_toggle_mode,
    ).pack(side=tk.RIGHT, padx=6, pady=4)

    save_status = tk.StringVar(value="")
    tk.Label(toolbar, textvariable=save_status,
             bg="#2a2a2a", fg="#aaaaaa", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=8)

    def _save_segmentation():
        if seg_mask[0] is None:
            save_status.set("セマセグ取得前です")
            return
        mask = seg_mask[0]
        if not mask.any():
            save_status.set("マスクが空です")
            return

        # マスク領域を RGBA で切り出す
        rgba = pil_image.convert("RGBA")
        arr  = np.array(rgba)
        arr[:, :, 3] = (mask.astype(np.uint8) * 255)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        cropped = Image.fromarray(arr).crop(
            (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
        )

        # result ディレクトリに保存
        result_dir = Path.cwd() / "result"
        result_dir.mkdir(exist_ok=True)
        stem = Path(path).stem
        i = 1
        while True:
            out_path = result_dir / f"{stem}_seg_{i:03d}.png"
            if not out_path.exists():
                break
            i += 1
        cropped.save(out_path)
        print(f"[Save] {out_path}")
        save_status.set(f"保存: {out_path.name}")

    tk.Button(
        toolbar, text="保存",
        bg="#226622", fg="white", relief=tk.FLAT, padx=10,
        command=_save_segmentation,
    ).pack(side=tk.LEFT, padx=6, pady=4)

    threading.Thread(target=_run_sam2, daemon=True).start()


# ---------------------------------------------------------------------------
# Results window
# ---------------------------------------------------------------------------

def open_results_window(root, results: list[dict], query_face_idx: int,
                        face_db=None, query_emb=None,
                        sam2_predictor=None, sam2_lock=None, yolo_model=None,
                        sam2_extra=None):
    win = tk.Toplevel(root)
    win.title(f"Search results for Face #{query_face_idx + 1}  ({len(results)} match(es))")
    win.configure(bg="#1e1e1e")
    win.state("normal")  # 最大化を防ぐ

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
        sim_pct   = rec["similarity"] * 100
        border_col = _sim_color(rec["similarity"])

        # 外枠（類似度カラー）→ 内枠（ダーク）の2層構造
        outer = tk.Frame(inner, bg=border_col, padx=3, pady=3)
        outer.grid(row=idx // RESULT_COLS, column=idx % RESULT_COLS, padx=6, pady=6, sticky="n")
        frame = tk.Frame(outer, bg="#2a2a2a")
        frame.pack()

        thumb = make_face_thumb(rec["pil_image"], rec["face"].bbox)
        photo = ImageTk.PhotoImage(thumb)
        photo_refs.append(photo)

        img_label = tk.Label(frame, image=photo, bg="#2a2a2a", cursor="hand2")
        img_label.pack()

        tk.Label(
            frame,
            text=f"{Path(rec['path']).name}\n{sim_pct:.1f}%",
            bg="#2a2a2a", fg="#cccccc", font=("Helvetica", 8),
            wraplength=THUMB_SIZE, justify=tk.CENTER,
        ).pack(pady=(2, 4))

        def open_full(event, r=rec):
            _open_full_image(root, r["pil_image"], r["face"].bbox, r["path"],
                             face_db=face_db, query_emb=query_emb,
                             sam2_predictor=sam2_predictor, sam2_lock=sam2_lock,
                             yolo_model=yolo_model, sam2_extra=sam2_extra)

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
        print("Usage: uv run face_finder.py <search_directory> [image_file]")
        sys.exit(1)

    search_dir = Path(sys.argv[1])
    image_path = sys.argv[2] if len(sys.argv) >= 3 else None

    if not search_dir.is_dir():
        print(f"Error: '{search_dir}' is not a directory")
        sys.exit(1)

    if image_path and not Path(image_path).is_file():
        print(f"Error: '{image_path}' is not a file")
        sys.exit(1)

    trt_cache_dir = Path.home() / ".cache" / "face_finder" / "trt"
    trt_cache_dir.mkdir(parents=True, exist_ok=True)

    trt_available  = "TensorrtExecutionProvider" in onnxruntime.get_available_providers()
    trt_needs_build = trt_available and not any(trt_cache_dir.glob("*.engine"))

    # --- ローディング画面（GUI を先に起動）---
    root = TkinterDnD.Tk()
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

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            # onnxruntime-gpu と PyTorch の CUDA コンテキスト競合を避けるため CPU で動作
            print("[Init] Loading SAM2 model (CPU)...")
            sam2 = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-small", device="cpu"
            )
            init_result["sam2_predictor"] = sam2
            print("[Init] SAM2 loaded (CPU)")
        except Exception as e:
            import traceback
            print(f"[Init] SAM2 not available: {e}")
            traceback.print_exc()
            init_result["sam2_predictor"] = None

        try:
            from ultralytics import YOLO
            print("[Init] Loading YOLOv8 Pose model (CPU)...")
            yolo = YOLO("yolov8n-pose.pt")
            yolo.to("cpu")
            init_result["yolo_model"] = yolo
            print("[Init] YOLOv8 Pose loaded (CPU)")
        except Exception as e:
            import traceback
            print(f"[Init] YOLOv8 Pose not available: {e}")
            traceback.print_exc()
            init_result["yolo_model"] = None

        root.after(0, _on_init_done)

    threading.Thread(target=_do_init, daemon=True).start()

    def _on_init_done():
        elapsed = time.time() - start_time
        print(f"[Init] Done in {elapsed:.1f}s")
        progress.stop()
        load_frame.destroy()

        if init_result.get("sam2_predictor") is None:
            from tkinter import messagebox
            messagebox.showerror(
                "初期化エラー",
                "SAM2 モデルの読み込みに失敗しました。\n"
                "ターミナルのエラーログを確認してください。"
            )
            root.destroy()
            return

        _setup_main(root, image_path, search_dir, init_result["face_app"],
                    sam2_predictor=init_result["sam2_predictor"],
                    yolo_model=init_result.get("yolo_model"))

    root.mainloop()


def _setup_main(root: tk.Tk, image_path: str | None, search_dir: Path, face_app,
                sam2_predictor=None, yolo_model=None):
    face_app_lock = threading.Lock()
    sam2_lock     = threading.Lock() if sam2_predictor is not None else None
    sam2_extra    = {"large": None}  # 高精度モデルの遅延ロード用
    INFO_PANEL_W  = 320
    PLACEHOLDER_W, PLACEHOLDER_H = 640, 480

    # スキャンを即時開始（search_dir は必須）
    face_db = DirectoryFaceDB(search_dir, face_app, face_app_lock)
    face_db.start_scan()

    # 現在表示中の画像ファイルパス（None = プレースホルダー表示中）
    current_path: list[Path | None] = [None]

    # 初期画像の顔検出
    if image_path:
        print("Detecting faces in main image...")
        pil_image = Image.open(image_path).convert("RGB")
        bgr_image = np.array(pil_image)[:, :, ::-1].copy()
        with face_app_lock:
            faces = face_app.get(bgr_image)
        print(f"Found {len(faces)} face(s)")
        current_path[0] = Path(image_path)
    else:
        pil_image = None
        faces     = []

    root.title("Face Finder" if not image_path else f"Face Finder - {image_path}")
    root.resizable(True, True)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    def _calc_scale(img: Image.Image) -> float:
        w, h = img.size
        return min((screen_w - INFO_PANEL_W) * 0.95 / w, screen_h * 0.95 / h, 1.0)

    if pil_image:
        scale  = _calc_scale(pil_image)
        init_w = int(pil_image.width  * scale)
        init_h = int(pil_image.height * scale)
    else:
        scale  = 1.0
        init_w = PLACEHOLDER_W
        init_h = PLACEHOLDER_H

    root.geometry(f"{init_w + INFO_PANEL_W}x{init_h}")

    # キャンバス
    canvas = tk.Canvas(root, width=init_w, height=init_h,
                       bg="#1e1e1e", cursor="crosshair", highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    box_items: list = []

    def _draw_image(img: Image.Image, sc: float):
        dw = int(img.width * sc)
        dh = int(img.height * sc)
        photo = ImageTk.PhotoImage(img.resize((dw, dh), Image.LANCZOS))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # GC防止
        return photo

    def _draw_placeholder():
        canvas.delete("all")
        canvas.config(width=PLACEHOLDER_W, height=PLACEHOLDER_H)
        cw, ch = PLACEHOLDER_W, PLACEHOLDER_H
        canvas.create_text(cw // 2, ch // 2,
                           text="画像をここにドロップしてください",
                           fill="#555555", font=("Helvetica", 18))

    if pil_image:
        _draw_image(pil_image, scale)
        for face in faces:
            x1, y1, x2, y2 = [c * scale for c in face.bbox]
            box_items.append(canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=2))
    else:
        _draw_placeholder()

    # 右パネル
    info_frame = tk.Frame(root, width=INFO_PANEL_W, bg="#1e1e1e")
    info_frame.pack(side=tk.RIGHT, fill=tk.Y)
    info_frame.pack_propagate(False)

    header_var = tk.StringVar(
        value=f"{len(faces)} face(s) detected\nClick to inspect / search"
              if pil_image else "画像をドロップして開始"
    )
    tk.Label(info_frame, textvariable=header_var, bg="#1e1e1e", fg="#cccccc",
             font=("Helvetica", 11), justify=tk.CENTER).pack(pady=(12, 6))

    info_text = scrolledtext.ScrolledText(
        info_frame, wrap=tk.WORD, font=("Courier", 9),
        bg="#2d2d2d", fg="#d4d4d4", insertbackground="white", state=tk.DISABLED,
    )
    info_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    status_var = tk.StringVar(value=f"Scanning {search_dir.name} ...")
    tk.Label(info_frame, textvariable=status_var, bg="#1e1e1e", fg="#888888",
             font=("Helvetica", 9)).pack(pady=4)

    def _poll_scan():
        if face_db.status == "done":
            status_var.set(f"Ready  ({len(face_db.records)} faces indexed)")
        else:
            if face_db.total > 0:
                status_var.set(f"Scanning {face_db.scanned}/{face_db.total} ...")
            root.after(300, _poll_scan)

    root.after(300, _poll_scan)

    # ------------------------------------------------------------------ helpers
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

    def _apply_image(new_path: str, new_pil: Image.Image, new_faces: list):
        nonlocal pil_image, faces, scale

        pil_image        = new_pil
        faces            = new_faces
        scale            = _calc_scale(new_pil)
        current_path[0]  = Path(new_path)
        dw = int(new_pil.width  * scale)
        dh = int(new_pil.height * scale)

        root.title(f"Face Finder - {new_path}")
        root.geometry(f"{dw + INFO_PANEL_W}x{dh}")
        canvas.config(width=dw, height=dh)
        _draw_image(new_pil, scale)

        box_items.clear()
        for face in faces:
            x1, y1, x2, y2 = [c * scale for c in face.bbox]
            box_items.append(canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=2))

        header_var.set(f"{len(faces)} face(s) detected\nClick to inspect / search")
        set_info("")

    # ------------------------------------------------------------------ drag & drop
    def _on_drop(event):
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            file_path = raw[1:-1]
        else:
            file_path = raw.split()[0]

        if Path(file_path).suffix.lower() not in IMAGE_EXTENSIONS:
            status_var.set("対応していないファイル形式です")
            return

        status_var.set("読み込み中 ...")
        print(f"[Drop] {file_path}")

        def _detect():
            try:
                new_pil = Image.open(file_path).convert("RGB")
                new_bgr = np.array(new_pil)[:, :, ::-1].copy()
                with face_app_lock:
                    new_faces = face_app.get(new_bgr)
                print(f"[Drop] Found {len(new_faces)} face(s)")
                root.after(0, _apply_image, file_path, new_pil, new_faces)
                root.after(0, status_var.set,
                           f"Ready  ({len(face_db.records)} faces indexed)"
                           if face_db.status == "done" else status_var.get())
            except Exception as e:
                print(f"[Drop] Error: {e}")
                root.after(0, status_var.set, f"エラー: {e}")

        threading.Thread(target=_detect, daemon=True).start()

    canvas.drop_target_register(DND_FILES)
    canvas.dnd_bind("<<Drop>>", _on_drop)

    # ------------------------------------------------------------------ click
    def on_click(event):
        if pil_image is None:
            return

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

        if face.embedding is not None:
            if face_db.status != "done":
                status_var.set("Still scanning — please wait...")
                return
            status_var.set("Searching...")
            query_emb = face.embedding.copy()

            # クエリ画像が検索ディレクトリ内にある場合は除外
            try:
                current_path[0].resolve().relative_to(search_dir.resolve())
                exclude = current_path[0]
            except (ValueError, AttributeError):
                exclude = None

            def do_search():
                results = face_db.search(query_emb, exclude_path=exclude)
                root.after(0, _show_results, results, clicked_idx, query_emb)

            threading.Thread(target=do_search, daemon=True).start()

    def _show_results(results, face_idx, query_emb):
        status_var.set(f"{len(results)} match(es) found")
        open_results_window(root, results, face_idx, face_db=face_db, query_emb=query_emb,
                            sam2_predictor=sam2_predictor, sam2_lock=sam2_lock,
                            yolo_model=yolo_model, sam2_extra=sam2_extra)

    canvas.bind("<Button-1>", on_click)


if __name__ == "__main__":
    main()
