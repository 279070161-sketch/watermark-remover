import base64
import io
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

from inpaint_engine import InpaintEngine

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

engine = InpaintEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure AI LaMa model starts downloading or loads in background
    engine.ensure_model_async()
    yield


app = FastAPI(title="AI 智能去水印工具", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/model-status")
async def model_status():
    if not engine.download_status["ready"] and not engine.download_status["downloading"]:
        engine.ensure_model_async()
    return JSONResponse(engine.download_status)


@app.post("/api/inpaint")
async def inpaint_endpoint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    mode: str = Form("lama"),
    dilate: int = Form(8),
):
    t0 = time.perf_counter()
    try:
        img_bytes = await image.read()
        mask_bytes = await mask.read()

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        if pil_mask.size != pil_img.size:
            pil_mask = pil_mask.resize(pil_img.size, Image.Resampling.NEAREST)

        img_np = np.array(pil_img, dtype=np.uint8)
        mask_np = np.array(pil_mask, dtype=np.uint8)

        dilate_px = max(0, min(30, int(dilate)))

        if mode == "lama":
            if not engine.download_status["ready"]:
                # Fallback or wait if nearly done
                raise HTTPException(
                    status_code=503,
                    detail="AI LaMa 模型正在后台下载中，请稍候或先切换至「OpenCV 极速修复」模式。",
                )
            res_np = engine.inpaint_lama(img_np, mask_np, dilate_px=dilate_px)
        elif mode == "opencv_ns":
            res_np = engine.inpaint_opencv(
                img_np, mask_np, dilate_px=dilate_px, radius=6, method="ns"
            )
        else:
            res_np = engine.inpaint_opencv(
                img_np, mask_np, dilate_px=dilate_px, radius=6, method="telea"
            )

        out_pil = Image.fromarray(res_np)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG", compress_level=3)
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return JSONResponse(
            {
                "success": True,
                "image_base64": f"data:image/png;base64,{b64_img}",
                "elapsed_ms": elapsed_ms,
                "mode": mode,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/demo-image")
async def get_demo_image():
    """Generate a rich landscape illustration with sample watermarks for instant testing."""
    w, h = 1024, 640
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Sky sunset gradient
    sky_r = np.clip(45 + 195 * (y_idx / h), 0, 255)
    sky_g = np.clip(65 + 115 * np.sin(np.pi * y_idx / h), 0, 255)
    sky_b = np.clip(150 - 70 * (y_idx / h) + 30 * (x_idx / w), 0, 255)
    canvas = np.stack([sky_r, sky_g, sky_b], axis=-1).astype(np.uint8)

    # Glowing sun
    sun_cx, sun_cy = int(w * 0.72), int(h * 0.32)
    dist = np.sqrt((x_idx - sun_cx) ** 2 + (y_idx - sun_cy) ** 2)
    sun_glow = np.exp(-dist / 85.0)[..., np.newaxis]
    sun_color = np.array([255, 235, 175], dtype=np.float32)
    canvas = np.clip(
        canvas.astype(np.float32) * (1 - 0.65 * sun_glow) + sun_color * 0.65 * sun_glow,
        0,
        255,
    ).astype(np.uint8)
    cv2.circle(canvas, (sun_cx, sun_cy), 42, (255, 248, 220), -1, lineType=cv2.LINE_AA)

    # Layered mountains
    mountain1_y = (
        h * 0.56
        - 55 * np.sin(x_idx[0] / 130.0)
        - 35 * np.cos(x_idx[0] / 52.0)
        - 18 * np.sin(x_idx[0] / 23.0)
    )
    for x in range(w):
        my = int(mountain1_y[x])
        canvas[my:int(h * 0.68), x] = [68, 72, 112]

    mountain2_y = (
        h * 0.63
        - 40 * np.cos((x_idx[0] + 100) / 110.0)
        - 25 * np.sin(x_idx[0] / 39.0)
    )
    for x in range(w):
        my = int(mountain2_y[x])
        canvas[my:int(h * 0.68), x] = [42, 48, 82]

    # Lake reflection with gentle ripples
    horizon = int(h * 0.68)
    for y in range(horizon, h):
        ref_y = max(0, horizon - int((y - horizon) * 1.15))
        shift = int(4.0 * np.sin(y * 0.45))
        row = np.roll(canvas[ref_y], shift, axis=0)
        canvas[y] = np.clip(row.astype(np.float32) * 0.82 + np.array([12, 22, 38]), 0, 255)

    # Convert to PIL to draw realistic platform watermarks
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Watermark 1: Bottom-right platform badge + text
    draw.rounded_rectangle(
        [w - 285, h - 78, w - 28, h - 24],
        radius=12,
        fill=(20, 24, 34, 155),
        outline=(255, 255, 255, 180),
        width=2,
    )
    draw.ellipse([w - 270, h - 66, w - 240, h - 36], fill=(255, 95, 85, 220))
    draw.text((w - 228, h - 65), "PHOTO STUDIO", fill=(255, 255, 255, 235))
    draw.text((w - 228, h - 46), "ID: @AI_WATERMARK_DEMO", fill=(210, 225, 255, 215))

    # Watermark 2: Top-left timestamp watermark
    draw.rounded_rectangle(
        [28, 26, 250, 68],
        radius=8,
        fill=(255, 255, 255, 125),
    )
    draw.text((42, 38), "REC  2026-11-03 14:30", fill=(30, 35, 45, 220))

    # Watermark 3: Floating semi-transparent text in sky
    draw.text((130, 185), "SAMPLE PREVIEW ONLY", fill=(255, 255, 255, 130))

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import sys
    import uvicorn

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    port = 8765
    print("\n=======================================================")
    print(" [OK] AI 智能去水印工具已启动！")
    print(f" [>>] 请在浏览器访问: http://127.0.0.1:{port}")
    print("=======================================================\n")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
