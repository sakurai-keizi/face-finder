# face_finder

画像内の顔をクリックし、指定ディレクトリ内の画像から同一人物を検索するGUIツールです。
顔特徴量の抽出・照合に [InsightFace](https://github.com/deepinsight/insightface) を使用します。

## 必要環境

- Python 3.10 以上
- [uv](https://github.com/astral-sh/uv)
- 依存パッケージは `uv run` 実行時に自動インストールされます（手動インストール不要）

GPU を使用する場合は別途システムへのインストールが必要です：

| 機能 | 必要なもの |
|------|-----------|
| CUDA（GPU推論） | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |
| TensorRT（高速GPU推論） | [TensorRT](https://developer.nvidia.com/tensorrt) + CUDA Toolkit |

GPU が無い環境では自動的に CPU にフォールバックします。

> **TensorRT について**: 初回起動時にモデルのエンジンビルドが走るため1〜数分かかります。2回目以降はキャッシュが使われるため高速に起動します。

## 使い方

### 基本（画像を開いて顔を確認するだけ）

```bash
uv run face_finder.py <画像ファイル>
```

### ディレクトリ内を検索

```bash
uv run face_finder.py <画像ファイル> <検索ディレクトリ>
```

**例:**

```bash
uv run face_finder.py group_photo.jpg ./photos/
```

## 操作方法

### メインウィンドウ

| 操作 | 動作 |
|------|------|
| 顔の緑枠内をクリック | 顔を選択（枠が黄色に変わる）、右パネルに特徴量を表示 |
| 枠外をクリック | 選択解除 |

右パネルには以下の情報が表示されます：

- バウンディングボックス座標
- 検出スコア
- 性別・推定年齢
- 特徴ベクトルの次元数・ノルム

### 検索ディレクトリ指定時

起動直後（GUI表示より前）からバックグラウンドでスキャンを開始します。
スキャン状況はターミナルとウィンドウ下部のステータスバーに表示されます。

```
[Cache] Loaded 8 image(s) / 218 face(s) from cache
[Scan] 8 image(s) found in 'photos'
[Scan] (1/8) img001.jpg  -> 40 face(s)  [cache]
[Scan] (2/8) img002.jpg  -> 3 face(s)
...
[Scan] Done. Total faces indexed: 218
[Cache] Saved to photos/.face_finder_cache.pkl
```

スキャン完了後に顔をクリックすると類似人物を検索し、**結果ウィンドウ**が開きます。

#### キャッシュ

スキャン済み画像の顔座標・特徴ベクトルは検索ディレクトリ内の `.face_finder_cache.pkl` に保存されます。次回起動時は画像バイナリの SHA-256 ハッシュで同一性を確認し、一致した場合は InsightFace の推論をスキップします。画像が変更された場合はハッシュが変わるため自動的に再検出されます。

### 結果ウィンドウ

- 画像ごとに最も類似度の高い顔を1件ずつ、類似度の高い順にグリッド表示
- 各サムネイルに **ファイル名** と **類似度（%）** を表示
- サムネイルをクリックすると元画像を別ウィンドウに表示
  - マッチした顔：**実線・黄色** BBOX
  - 同画像内の他の顔で類似度 ≥ 0.20 の顔：**点線・オレンジ** BBOX
- マウスホイールでスクロール可能

## 対応画像フォーマット

`.jpg` / `.jpeg` / `.png` / `.bmp` / `.webp` / `.tiff` / `.tif`

## 設定値（スクリプト先頭の定数）

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `SIMILARITY_THRESHOLD` | `0.35` | 同一人物と判定するコサイン類似度の閾値（0〜1） |
| `PLAUSIBLE_THRESHOLD` | `0.20` | 元画像で点線BBOXを付ける下限閾値 |
| `THUMB_SIZE` | `220` | 結果サムネイルのサイズ（px） |
| `RESULT_COLS` | `4` | 結果グリッドの列数 |

類似度の閾値は顔の明瞭さや照明条件によって調整が必要な場合があります。値を上げると厳しく（誤検出が減る）、下げると緩く（見逃しが減る）なります。

## 技術詳細

- **顔検出・特徴抽出**: InsightFace `buffalo_l` モデル（512次元 embedding）
- **類似度計算**: コサイン類似度
- **推論バックエンド**: TensorRT → CUDA → CPU の順で自動選択
- **GUI**: tkinter（標準ライブラリ）
- **スキャン**: 起動直後からバックグラウンドスレッドで実行（GUI操作をブロックしない）
- **キャッシュ**: SHA-256ハッシュをキーにした pickle ファイル（`.face_finder_cache.pkl`）
