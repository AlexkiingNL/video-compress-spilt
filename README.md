# 🎥 Smart Video Compressor & Splitter

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

A professional-grade video compression tool designed for **macOS**. It leverages Apple's `hevc_videotoolbox` for hardware-accelerated compression and includes an intelligent auto-splitting feature for file size constraints.

## 💡 Motivation

Many platforms impose strict file size limits on uploads. For example, **Google NotebookLM** only supports files smaller than **200MB**. This tool was created specifically to address this limitation.

It employs a **"Quality-First" Smart Strategy**:
1.  **Adaptive Compression**: It first attempts to compress the video bitrate and audio parameters to fit within the target size (Default: 200MB) while maintaining high clarity.
2.  **Minimal Splitting**: If the compressed file still exceeds the target, the tool **will not sacrifice quality** further. Instead, it calculates the minimum number of segments required and splits the video, preventing redundant fragmentation.

> **Tip**: For strict limits (like 200MB), it is recommended to set a safety margin, e.g., `-s 180`.

*Note: This tool is optimized for macOS users out-of-the-box. Users on other operating systems may need to modify the hardware encoder settings in the source code.*

## 🚀 Key Features

- **Hardware Acceleration**: Uses Apple Silicon/Intel `Videotoolbox` for high-speed HEVC encoding.
- **Smart Splitting**: Automatically splits large videos into parts (e.g., <200MB) *after* compression to fit platform limits (Discord, WeChat, etc.).
- **Batch Processing**: Recursively scans directories for video files.
- **Data-Driven**: Analyzes bitrate and adapts compression parameters dynamically.
- **Visual Feedback**: Beautiful terminal UI using `rich`.

## 🛠 Prerequisites

- **macOS** (Required for `hevc_videotoolbox`)
- **Python 3.8+**
- **FFmpeg**: Must be installed and accessible in system PATH.

```bash
brew install ffmpeg
pip install -r requirements.txt

📖 Usage
Basic Compression
Compress a single file or a directory of videos to the default ./compressed folder.
Bash
python3 smart_video_compressor_splitter.py ./my_videos/
Custom Split Size
Compress and ensure no file exceeds 100MB.
Bash
python3 smart_video_compressor_splitter.py ./movie.mp4 -s 100
Custom Output Directory
Bash
python3 smart_video_compressor_splitter.py ./input -o ./final_output

## 🐧 For Windows / Linux Users (Experimental)

This tool is natively optimized for macOS. However, the core logic (FFmpeg + Python) is cross-platform. If you wish to run this on Windows (NVIDIA GPU) or Linux, you can manually modify the source code:

1.  Open `smart_video_compressor_splitter.py`.
2.  Search for the `CompressorEngine` class.
3.  Replace the encoder `hevc_videotoolbox` with one supported by your hardware:
    * **NVIDIA GPU**: `hevc_nvenc` or `h264_nvenc`
    * **CPU (Universal)**: `libx264` (Slower, but works everywhere)

```python
# Example modification for CPU-only (Universal compatibility)
cmd = [
    "ffmpeg", "-y", "-i", str(input_path),
    ...
    "-c:v", "libx264",  # <--- Change this line
    ...
]

Note: Windows users may also need to ensure ffmpeg is added to their System PATH.

📄 License
This project is licensed under the MIT License. 
