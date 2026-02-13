#!/usr/bin/env python3 
"""
Smart Video Compressor & Splitter
---------------------------------
A professional tool to compress videos using hardware acceleration (macOS)
and automatically split them into parts based on a target file size.

Author: Open Source Community
License: MIT
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TaskID
    )
    from rich.table import Table
except ImportError:
    print("Error: 'rich' library is required. Please install it via 'pip install rich'.")
    sys.exit(1)

# Initialize Console
console = Console()

# Constants
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}
DEFAULT_TARGET_SIZE_MB = 200
DEFAULT_OUTPUT_DIR = os.getenv("VIDEO_OUTPUT_DIR", "./compressed")


class VideoProbe:
    """Handles video metadata extraction using ffprobe or ffmpeg."""

    @staticmethod
    def _probe_with_ffprobe(filepath: Path) -> Optional[Dict[str, Any]]:
        """Extracts metadata using ffprobe (Preferred method)."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        fmt = data.get("format", {})
        streams = data.get("streams", [])

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if video_stream is None:
            return None

        try:
            duration = float(fmt.get("duration", 0))
            file_size = int(fmt.get("size", 0))
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))

            video_bitrate = int(video_stream.get("bit_rate", 0))
            # Fallback bitrate calculation
            if video_bitrate == 0:
                total_bitrate = int(fmt.get("bit_rate", 0))
                audio_br = int(audio_stream.get("bit_rate", 0)) if audio_stream else 0
                video_bitrate = max(total_bitrate - audio_br, 0)

            audio_codec = audio_stream.get("codec_name", "unknown") if audio_stream else "none"
            audio_bitrate = int(audio_stream.get("bit_rate", 0)) if audio_stream else 0

            return {
                "duration": duration,
                "width": width,
                "height": height,
                "video_bitrate": video_bitrate,
                "audio_codec": audio_codec,
                "audio_bitrate": audio_bitrate,
                "file_size": file_size,
            }
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _probe_with_ffmpeg(filepath: Path) -> Optional[Dict[str, Any]]:
        """Extracts metadata using ffmpeg -i (Fallback method)."""
        cmd = ["ffmpeg", "-i", str(filepath)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr

        # Regex parsing for fallback
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
        if not dur_match:
            return None
        
        duration = (int(dur_match.group(1)) * 3600 +
                    int(dur_match.group(2)) * 60 +
                    float(dur_match.group(3)))

        total_bitrate = 0
        br_match = re.search(r"Duration:.*?bitrate:\s*(\d+)\s*kb/s", output)
        if br_match:
            total_bitrate = int(br_match.group(1)) * 1000

        file_size = os.path.getsize(filepath)

        width, height, video_bitrate = 0, 0, 0
        video_match = re.search(r"Stream\s+#\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5})", output)
        if video_match:
            width = int(video_match.group(1))
            height = int(video_match.group(2))

        vid_br_match = re.search(r"Stream\s+#\d+:\d+.*?Video:.*?(\d+)\s*kb/s", output)
        if vid_br_match:
            video_bitrate = int(vid_br_match.group(1)) * 1000

        audio_codec = "none"
        audio_bitrate = 0
        audio_match = re.search(r"Stream\s+#\d+:\d+.*?Audio:\s*(\w+)", output)
        if audio_match:
            audio_codec = audio_match.group(1).lower()

        aud_br_match = re.search(r"Stream\s+#\d+:\d+.*?Audio:.*?(\d+)\s*kb/s", output)
        if aud_br_match:
            audio_bitrate = int(aud_br_match.group(1)) * 1000

        if video_bitrate == 0 and total_bitrate > 0:
            video_bitrate = max(total_bitrate - audio_bitrate, 0)

        return {
            "duration": duration,
            "width": width,
            "height": height,
            "video_bitrate": video_bitrate,
            "audio_codec": audio_codec,
            "audio_bitrate": audio_bitrate,
            "file_size": file_size,
        }

    @classmethod
    def probe(cls, filepath: Path) -> Dict[str, Any]:
        """Main entry point for probing video information."""
        if shutil.which("ffprobe"):
            info = cls._probe_with_ffprobe(filepath)
            if info:
                return info
        
        info = cls._probe_with_ffmpeg(filepath)
        if info:
            return info
            
        raise RuntimeError(f"Unable to probe video information: {filepath}")


class CompressorEngine:
    """Handles the compression logic and ffmpeg execution."""

    @staticmethod
    def compute_params(info: Dict[str, Any]) -> Dict[str, Union[int, str]]:
        """Calculates optimal compression parameters."""
        orig_video_kbps = info["video_bitrate"] / 1000 if info["video_bitrate"] > 0 else 1500
        # Cap bitrate at 1500kbps to ensure size reduction, min 100kbps
        video_kbps = max(100, min(int(orig_video_kbps), 1500))

        orig_audio_kbps = info["audio_bitrate"] / 1000 if info["audio_bitrate"] > 0 else 0
        
        if info["audio_codec"] == "aac" and orig_audio_kbps <= 128:
            audio_mode = "copy"
            audio_kbps = orig_audio_kbps
        elif info["audio_codec"] == "none":
            audio_mode = "none"
            audio_kbps = 0
        else:
            audio_mode = "transcode"
            audio_kbps = 128

        return {
            "video_kbps": video_kbps,
            "audio_kbps": audio_kbps,
            "audio_mode": audio_mode,
        }

    @staticmethod
    def check_hw_encoder() -> bool:
        """Checks if hevc_videotoolbox (macOS hardware acceleration) is available."""
        cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "hevc_videotoolbox" in result.stdout

    @staticmethod
    def compress(input_path: Path, output_path: Path, duration: float, 
                 info: Dict[str, Any], params: Dict[str, Any], 
                 progress: Progress, task_id: TaskID) -> bool:
        """Executes the ffmpeg compression command."""
        video_kbps = params["video_kbps"]
        audio_mode = params["audio_mode"]
        # Downscale to 720p if original is larger
        scale_filter = "scale=-2:720" if info["height"] > 720 else None

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-progress", "pipe:1", "-nostats",
            "-c:v", "hevc_videotoolbox",
            "-b:v", f"{video_kbps}k",
            "-maxrate", f"{int(video_kbps * 1.2)}k",
            "-bufsize", f"{int(video_kbps * 2)}k",
        ]
        
        if scale_filter:
            cmd += ["-vf", scale_filter]

        if audio_mode == "copy":
            cmd += ["-c:a", "copy"]
        elif audio_mode == "none":
            cmd += ["-an"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "128k"]

        cmd += ["-tag:v", "hvc1", "-movflags", "+faststart", str(output_path)]
        
        return CompressorEngine._run_ffmpeg_with_progress(cmd, duration, progress, task_id) == 0

    @staticmethod
    def _run_ffmpeg_with_progress(cmd: List[str], duration: float, 
                                  progress: Progress, task_id: TaskID) -> int:
        """Runs ffmpeg and parses stdout for progress bars."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        for line in process.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=")[1])
                    current_secs = us / 1_000_000
                    progress.update(task_id, completed=min(current_secs, duration))
                except (ValueError, IndexError):
                    pass

        process.wait()
        return process.returncode

    @staticmethod
    def split_by_size(input_path: Path, target_size_mb: int, 
                      output_dir: Path, stem: str) -> List[Path]:
        """Splits video into parts based on target size."""
        info = VideoProbe.probe(input_path)
        file_size_mb = info["file_size"] / (1024 * 1024)

        num_parts = max(1, math.ceil(file_size_mb / target_size_mb))
        
        # Optimize split count to avoid tiny last parts
        if num_parts > 1:
            avg_part_mb = file_size_mb / num_parts
            last_part_mb = file_size_mb - avg_part_mb * (num_parts - 1)
            if last_part_mb < target_size_mb * 0.85 and num_parts > 2:
                num_parts -= 1

        segment_duration = info["duration"] / num_parts
        pattern = str(output_dir / f"{stem}_part%d.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-reset_timestamps", "1",
            "-avoid_negative_ts", "make_zero",
            "-segment_start_number", "1",
            pattern,
        ]
        subprocess.run(cmd, capture_output=True)

        return sorted(output_dir.glob(f"{stem}_part*.mp4"))


class Utils:
    """Utility functions for file handling and formatting."""

    @staticmethod
    def format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / 1024 / 1024:.1f}MB"
        else:
            return f"{size_bytes / 1024 / 1024 / 1024:.2f}GB"

    @staticmethod
    def format_duration(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @staticmethod
    def collect_videos(input_path: Path) -> List[Path]:
        path_obj = Path(input_path)
        if path_obj.is_file():
            if path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
                return [path_obj]
            else:
                console.print(f"[red]Unsupported file format: {path_obj.suffix}[/red]")
                return []
        elif path_obj.is_dir():
            videos = []
            for ext in SUPPORTED_EXTENSIONS:
                videos.extend(path_obj.rglob(f"*{ext}"))
                videos.extend(path_obj.rglob(f"*{ext.upper()}"))
            
            # Deduplicate and sort
            seen = set()
            unique = []
            for v in sorted(videos):
                resolved = v.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    unique.append(v)
            return unique
        else:
            console.print(f"[red]Path does not exist: {input_path}[/red]")
            return []


def main():
    parser = argparse.ArgumentParser(
        description="Smart Video Compressor & Splitter",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("-s", "--size", type=int, default=DEFAULT_TARGET_SIZE_MB,
                        help=f"Target size per part in MB (Default: {DEFAULT_TARGET_SIZE_MB})")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, 
                        help=f"Output directory (Default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="Show compression plan only")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    target_size_mb = args.size
    dry_run = args.dry_run

    if not CompressorEngine.check_hw_encoder():
        console.print("[red]✗ Error: 'hevc_videotoolbox' (macOS) not available.[/red]")
        sys.exit(1)
    console.print("[green]✓ Hardware acceleration (hevc_videotoolbox) detected[/green]")

    videos = Utils.collect_videos(input_path)
    if not videos:
        console.print("[red]No valid video files found.[/red]")
        sys.exit(1)

    console.print(f"\nFound [bold]{len(videos)}[/bold] video files\n")

    # Parallel Probing
    video_infos = []
    
    def _probe_wrapper(v):
        try:
            return v, VideoProbe.probe(v)
        except RuntimeError as e:
            return v, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(videos), 8)) as pool:
        future_map = {pool.submit(_probe_wrapper, v): v for v in videos}
        for future in concurrent.futures.as_completed(future_map):
            v, info = future.result()
            if info:
                video_infos.append((v, info))
            else:
                console.print(f"  [red]Skipping {v.name}: Unable to probe metadata[/red]")

    # Restore order
    video_order = {v: i for i, v in enumerate(videos)}
    video_infos.sort(key=lambda x: video_order[x[0]])

    if not video_infos:
        sys.exit(1)

    # Planning Phase
    plans = []
    for v, info in video_infos:
        params = CompressorEngine.compute_params(info)
        total_kbps = params["video_kbps"] + params["audio_kbps"]
        est_compressed_mb = total_kbps * info["duration"] / 8 / 1024
        need_split = est_compressed_mb > target_size_mb
        est_parts = max(1, math.ceil(est_compressed_mb / target_size_mb)) if need_split else 1
        plans.append((v, info, params, est_compressed_mb, need_split, est_parts))

    # Display Plan
    table = Table(title=f"Compression Plan (Target: ≤ {target_size_mb}MB/part)")
    table.add_column("Filename", style="cyan", no_wrap=True, max_width=50)
    table.add_column("Duration", justify="right")
    table.add_column("Original Size", justify="right")
    table.add_column("Video Bitrate", justify="right")
    table.add_column("Est. Size", justify="right")
    table.add_column("Segments", justify="center")

    for v, info, params, est_mb, need_split, est_parts in plans:
        split_str = f"[bold]{est_parts}[/bold]" if need_split else "[green]No[/green]"
        table.add_row(
            v.name,
            Utils.format_duration(info["duration"]),
            Utils.format_size(info["file_size"]),
            f"{params['video_kbps']}kbps",
            f"~{est_mb:.0f}MB",
            split_str,
        )

    console.print(table)
    console.print(
        f"\nEncoder: [bold]hevc_videotoolbox[/bold]  |  "
        f"Resolution: [bold]720p[/bold]  |  "
        f"Audio: [bold]AAC 128k[/bold]\n"
    )

    if dry_run:
        console.print("[yellow]Dry-run mode active. No changes made.[/yellow]")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execution Phase
    results = []
    total_start = time.time()
    total_videos = len(plans)
    temp_dir = tempfile.mkdtemp(prefix="videocompress_")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task(f"Total Progress", total=total_videos)

            for idx, (v, info, params, est_mb, need_split, est_parts) in enumerate(plans):
                progress.update(overall_task, description=f"Processing ({idx + 1}/{total_videos})")
                console.print(f"\n[bold]▶ {v.name}[/bold]")

                # Check if copy is sufficient
                orig_size_mb = info["file_size"] / (1024 * 1024)
                if orig_size_mb <= target_size_mb:
                    out_file = output_dir / v.name
                    shutil.copy2(str(v), str(out_file))
                    console.print(f"  [green]✓ Original size within limits. Copied directly.[/green]")
                    results.append({
                        "name": out_file.name,
                        "input": info["file_size"], "output": info["file_size"],
                        "ratio": 0.0, "time": 0.0, "status": "COPIED"
                    })
                    progress.update(overall_task, advance=1)
                    continue

                # Compression
                compressed_tmp = Path(temp_dir) / f"{v.stem}_compressed.mp4"
                file_task = progress.add_task(f"  Compressing", total=info["duration"])
                
                start_t = time.time()
                success = CompressorEngine.compress(
                    v, compressed_tmp, info["duration"],
                    info, params, progress, file_task,
                )
                elapsed = time.time() - start_t
                progress.update(file_task, visible=False) # Hide finished task

                if not success or not compressed_tmp.exists():
                    console.print(f"  [red]✗ Compression failed[/red]")
                    results.append({
                        "name": v.name, "input": info["file_size"], "output": 0,
                        "ratio": 0.0, "time": elapsed, "status": "FAILED"
                    })
                    progress.update(overall_task, advance=1)
                    continue

                compressed_size = os.path.getsize(compressed_tmp)
                ratio = (1 - compressed_size / info["file_size"]) * 100
                console.print(
                    f"  [green]✓ Compression Complete[/green]: "
                    f"{Utils.format_size(info['file_size'])} -> {Utils.format_size(compressed_size)} "
                    f"(-{ratio:.1f}%) in {Utils.format_duration(elapsed)}"
                )

                # Split if necessary
                compressed_mb = compressed_size / (1024 * 1024)
                if compressed_mb <= target_size_mb:
                    out_file = output_dir / f"{v.stem}_compressed.mp4"
                    shutil.move(str(compressed_tmp), str(out_file))
                    results.append({
                        "name": out_file.name,
                        "input": info["file_size"], "output": compressed_size,
                        "ratio": ratio, "time": elapsed, "status": "SUCCESS"
                    })
                else:
                    console.print(f"  [yellow]Size ({compressed_mb:.0f}MB) exceeds target. Splitting...[/yellow]")
                    parts = CompressorEngine.split_by_size(
                        compressed_tmp, target_size_mb, output_dir, v.stem
                    )
                    
                    for p in parts:
                        results.append({
                            "name": p.name,
                            "input": info["file_size"], "output": os.path.getsize(p),
                            "ratio": ratio, "time": elapsed / len(parts), "status": "SPLIT"
                        })
                    compressed_tmp.unlink()

                progress.update(overall_task, advance=1)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary Report
    console.print("\n")
    summary = Table(title="Processing Summary")
    summary.add_column("Filename", style="cyan")
    summary.add_column("Status", justify="center")
    summary.add_column("Final Size", justify="right")
    summary.add_column("Time", justify="right")

    for r in results:
        status_color = "green" if r["status"] in ["SUCCESS", "SPLIT", "COPIED"] else "red"
        summary.add_row(
            r["name"],
            f"[{status_color}]{r['status']}[/{status_color}]",
            Utils.format_size(r["output"]),
            Utils.format_duration(r["time"])
        )
    
    console.print(summary)
    console.print(f"\n[bold]Output Directory:[/bold] {output_dir.resolve()}")

if __name__ == "__main__":
    main()
