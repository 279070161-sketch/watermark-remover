# ✨ 极简去水印工具 (AI LaMa + 纯前端双模版)

一款极简、好用的图片去水印工具，支持**框选即自动去水印**、**涂抹去水印**、**Ctrl+V 粘贴截图**与**一键复制保存**。

同时支持两种运行方式：
1. **本地/云端 AI 增强模式（Python + LaMa 深度学习模型）**：适合复杂背景无痕修复。
2. **纯前端无服务器在线模式（GitHub Pages / 单文件直开）**：无需安装任何环境，部署到 GitHub Pages 即可在线使用，或直接双击 `docs/index.html` 打开即用。

---

## 🚀 方式一：部署到 GitHub Pages 变成在线网站（零成本、永久免费）

1. 将本项目上传到你的 GitHub 仓库。
2. 进入 GitHub 仓库的 **Settings** -> **Pages**。
3. 在 **Build and deployment** 下：
   - **Source** 选择 `Deploy from a branch`
   - **Branch** 选择 `main`（或 `master`），文件夹选择 `/docs`，点击 **Save**。
4. 等待约 1 分钟，即可通过 `https://<你的GitHub用户名>.github.io/<仓库名>/` 在线访问使用！

> 💡 **单文件离线使用**：你也可以直接用浏览器双击打开 [`docs/index.html`](./docs/index.html)，无需启动 Python 也能直接使用纯前端引擎去水印。

---

## 🧠 方式二：本地启动 AI LaMa 深度学习版

1. Windows 下直接双击运行 **`启动去水印工具.bat`**。
2. 或使用命令行启动：
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
3. 浏览器会自动打开 `http://127.0.0.1:8765`（首次运行若无本地模型会自动下载约 198MB 的 `lama_fp32.onnx` 模型到 `models/` 目录）。
