import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import cv2
import numpy as np
import onnxruntime as ort
import requests

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "lama_fp32.onnx"

MODEL_URLS = [
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx",
]


class InpaintEngine:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.lock = threading.Lock()
        self.download_status = {
            "downloading": False,
            "progress": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "ready": False,
            "error": None,
        }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if self._is_model_valid():
            self.download_status["ready"] = True
            self.download_status["progress"] = 100.0

    def _is_model_valid(self) -> bool:
        return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 50 * 1024 * 1024

    def ensure_model_async(self):
        """Start background download of the LaMa ONNX model if not already present."""
        if self._is_model_valid():
            self.download_status["ready"] = True
            self.download_status["progress"] = 100.0
            return
        if self.download_status["downloading"]:
            return
        thread = threading.Thread(target=self._download_model, daemon=True)
        thread.start()

    def _download_model(self):
        self.download_status["downloading"] = True
        self.download_status["error"] = None
        temp_path = MODEL_PATH.with_suffix(".onnx.part")

        for url in MODEL_URLS:
            try:
                with requests.get(url, stream=True, timeout=20) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length", 0))
                    self.download_status["total_mb"] = round(total / (1024 * 1024), 1)
                    downloaded = 0
                    with open(temp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 512):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                self.download_status["downloaded_mb"] = round(
                                    downloaded / (1024 * 1024), 1
                                )
                                if total > 0:
                                    self.download_status["progress"] = round(
                                        downloaded * 100.0 / total, 1
                                    )
                if temp_path.exists() and temp_path.stat().st_size > 50 * 1024 * 1024:
                    temp_path.replace(MODEL_PATH)
                    self.download_status["ready"] = True
                    self.download_status["progress"] = 100.0
                    self.download_status["downloading"] = False
                    # Pre-load session
                    self.get_session()
                    return
            except Exception as e:
                self.download_status["error"] = str(e)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

        self.download_status["downloading"] = False

    def get_session(self) -> ort.InferenceSession:
        with self.lock:
            if self.session is None:
                if not self._is_model_valid():
                    self._download_model()
                if not self._is_model_valid():
                    raise RuntimeError(
                        f"AI 模型尚未就绪: {self.download_status.get('error') or '正在下载中'}"
                    )
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(
                    str(MODEL_PATH), sess_options=sess_options, providers=providers
                )
            return self.session

    def _cluster_mask_boxes(
        self, mask: np.ndarray, merge_distance: int = 160
    ) -> List[Tuple[int, int, int, int]]:
        """
        Group mask connected components into bounding boxes (x1, y1, x2, y2).
        Components closer than `merge_distance` are merged so nearby watermarks
        get repaired together in a single 512x512 context pass.
        """
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8
        )
        boxes: List[Tuple[int, int, int, int]] = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 0:
                boxes.append((int(x), int(y), int(x + w), int(y + h)))

        if not boxes:
            return []

        merged = True
        while merged:
            merged = False
            new_boxes: List[Tuple[int, int, int, int]] = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                bx1, by1, bx2, by2 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    ox1, oy1, ox2, oy2 = boxes[j]
                    # Check if boxes overlap when expanded by merge_distance
                    if not (
                        ox1 > bx2 + merge_distance
                        or ox2 < bx1 - merge_distance
                        or oy1 > by2 + merge_distance
                        or oy2 < by1 - merge_distance
                    ):
                        # Only merge if combined box isn't excessively huge unless they truly overlap
                        comb_w = max(bx2, ox2) - min(bx1, ox1)
                        comb_h = max(by2, oy2) - min(by1, oy1)
                        if comb_w <= 440 and comb_h <= 440:
                            bx1, by1 = min(bx1, ox1), min(by1, oy1)
                            bx2, by2 = max(bx2, ox2), max(by2, oy2)
                            used[j] = True
                            merged = True
                used[i] = True
                new_boxes.append((bx1, by1, bx2, by2))
            boxes = new_boxes

        return boxes

    def _run_lama_512(self, img_rgb_512: np.ndarray, mask_512: np.ndarray) -> np.ndarray:
        """
        Run LaMa ONNX on a 512x512 RGB image and 512x512 binary mask.
        Returns a 512x512 uint8 RGB image.
        """
        session = self.get_session()

        img_norm = img_rgb_512.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 512, 512)

        mask_bin = (mask_512 > 127).astype(np.float32)
        mask_chw = mask_bin[np.newaxis, np.newaxis, ...]  # (1, 1, 512, 512)

        outputs = session.run(None, {"image": img_chw, "mask": mask_chw})
        pred = outputs[0][0]  # (3, 512, 512)
        if pred.max() <= 1.5:
            pred = pred * 255.0
        pred = np.clip(pred, 0, 255).astype(np.uint8)
        pred_hwc = np.transpose(pred, (1, 2, 0))
        return pred_hwc

    def inpaint_lama(
        self, image_rgb: np.ndarray, mask_gray: np.ndarray, dilate_px: int = 8
    ) -> np.ndarray:
        """
        Smart patch-based LaMa inpainting:
        - Dilates mask slightly to cover watermark JPEG halos
        - Crops context windows around watermark regions (preserving 100% native resolution!)
        - Feather-blends repaired pixels seamlessly into the original image
        """
        h, w = image_rgb.shape[:2]
        binary_mask = (mask_gray > 64).astype(np.uint8) * 255
        if np.count_nonzero(binary_mask) == 0:
            return image_rgb.copy()

        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            dilated_mask = binary_mask

        boxes = self._cluster_mask_boxes(dilated_mask)
        result = image_rgb.copy()

        for bx1, by1, bx2, by2 in boxes:
            # Skip if this region was already repaired inside a previous crop window
            if np.count_nonzero(dilated_mask[by1:by2, bx1:bx2]) == 0:
                continue

            bw = bx2 - bx1
            bh = by2 - by1
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2

            # Determine crop size: minimum 512 (for 1:1 native pixel resolution),
            # or larger with context margin if the watermark region exceeds ~400px
            crop_size = max(512, int(max(bw, bh) * 1.4))

            # If crop_size <= min(w, h), we can extract an exact square inside the image
            x1 = cx - crop_size // 2
            y1 = cy - crop_size // 2
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            # Shift window to stay within image bounds if possible
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x2 > w:
                x1 = max(0, x1 - (x2 - w))
                x2 = w
            if y2 > h:
                y1 = max(0, y1 - (y2 - h))
                y2 = h

            img_crop = result[y1:y2, x1:x2]
            mask_crop = dilated_mask[y1:y2, x1:x2].copy()
            # Mark these mask pixels as repaired so subsequent boxes don't re-inpaint them
            dilated_mask[y1:y2, x1:x2] = 0

            ch, cw = img_crop.shape[:2]

            # If the image itself is smaller than 512 in either dimension, pad with reflection
            pad_b = max(0, 512 - ch)
            pad_r = max(0, 512 - cw)
            if pad_b > 0 or pad_r > 0:
                img_padded = cv2.copyMakeBorder(
                    img_crop, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101
                )
                mask_padded = cv2.copyMakeBorder(
                    mask_crop, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0
                )
            else:
                img_padded = img_crop
                mask_padded = mask_crop

            ph, pw = img_padded.shape[:2]
            if ph != 512 or pw != 512:
                img_512 = cv2.resize(img_padded, (512, 512), interpolation=cv2.INTER_AREA)
                mask_512 = cv2.resize(
                    mask_padded, (512, 512), interpolation=cv2.INTER_NEAREST
                )
                repaired_512 = self._run_lama_512(img_512, mask_512)
                repaired_padded = cv2.resize(
                    repaired_512, (pw, ph), interpolation=cv2.INTER_CUBIC
                )
            else:
                repaired_padded = self._run_lama_512(img_padded, mask_padded)

            repaired_crop = repaired_padded[:ch, :cw]

            # Feather mask edges for seamless compositing
            feather = cv2.GaussianBlur(
                mask_crop.astype(np.float32) / 255.0, (7, 7), 2.0
            )[..., np.newaxis]
            blended = (
                repaired_crop.astype(np.float32) * feather
                + img_crop.astype(np.float32) * (1.0 - feather)
            )
            result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result

    def inpaint_opencv(
        self,
        image_rgb: np.ndarray,
        mask_gray: np.ndarray,
        dilate_px: int = 5,
        radius: int = 5,
        method: str = "telea",
    ) -> np.ndarray:
        """
        Fast OpenCV fluid/telea inpainting (great for small text/thin lines or instant preview).
        """
        binary_mask = (mask_gray > 64).astype(np.uint8) * 255
        if np.count_nonzero(binary_mask) == 0:
            return image_rgb.copy()

        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            dilated_mask = binary_mask

        flags = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        repaired_bgr = cv2.inpaint(bgr, dilated_mask, inpaintRadius=radius, flags=flags)
        return cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)
