#!/usr/bin/env python3
"""Export the actual app recording to MP4 and GIF, without added text."""
from pathlib import Path
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/assets/demo'
WORK = ROOT / '.private/media-tools'
MEDIA.mkdir(parents=True, exist_ok=True)
info = json.loads((WORK / 'capture-info.json').read_text())
raw = WORK / 'capture.webm'
start = max(0.0, info.get('video_in_seconds', 0.0))
# Trim initial loading/setup only. Preserve real-time execution and interactions.
subprocess.run([
    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
    '-ss', str(start), '-i', str(raw), '-an',
    '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x172033',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '21',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
    str(MEDIA / 'packing-demo.mp4'),
], check=True)
# Both names point to the same clean publication cut, with no title cards.
shutil.copy2(MEDIA / 'packing-demo.mp4', MEDIA / 'linkedin-demo.mp4')
preview_start = info.get('solved_video_seconds', start + 8) + 0.4
subprocess.run([
    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
    '-ss', str(preview_start), '-t', '9', '-i', str(raw),
    '-filter_complex', 'fps=10,scale=880:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
    '-loop', '0', str(MEDIA / 'packing-demo.gif'),
], check=True)
subprocess.run([
    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
    '-i', str(MEDIA / 'linkedin-demo.mp4'),
    '-vf', 'fps=1/5,scale=480:-2,tile=3x2', '-frames:v', '1',
    str(WORK / 'contact-sheet.png'),
], check=True)
print(json.dumps({'files': [{'name': p.name, 'bytes': p.stat().st_size}
                           for p in sorted(MEDIA.iterdir()) if p.is_file()]}, indent=2))
