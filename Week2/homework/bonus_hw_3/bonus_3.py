import json
import re
from dataclasses import dataclass, asdict
from typing import List

import yt_dlp
import whisper
from yt_dlp.utils import download_range_func
import torch

if torch.cuda.is_available():
    model = whisper.load_model("base", device="cuda")
    print("✅ Using CUDA (GPU)")
elif torch.backends.mps.is_available():
    model = whisper.load_model("base", device="mps")
    print("🟡 Using MPS (Apple Silicon GPU)")
else:
    model = whisper.load_model("base")
    print("🔴 Using CPU")

@dataclass
class Segment:
    start: float
    end:   float
    text:  str

@dataclass
class TalkTranscript:
    video_name: str
    segments: List[Segment]

def save_all_transcripts_to_jsonl(talks, output_path="talks_transcripts.jsonl"):
    """
    talks: List of TalkTranscript
    """
    with open(output_path, "w", encoding="utf-8") as fout:
        for talk in talks:
            # Convert dataclasses to dicts for JSON serialization
            obj = {
                "video_name": talk.video_name,
                "segments": [asdict(seg) for seg in talk.segments]
            }
            fout.write(json.dumps(obj) + "\n")

def transcribe_with_whisper(video_path):
    result = model.transcribe(video_path, language="en")
    return result['segments']

def get_video_title(url):
    ydl_opts = {'quiet': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('title', 'unknown_title')

def sanitize_filename(name):
    # Remove or replace characters that are invalid for filenames
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# section download because of bandwidth/space save reasons
def download_section(url, output_path, start=None, end=None, full_vid=False):
    ydl_opts = {
        'quiet': True,
        'verbose': False,
        'outtmpl': output_path,
        'force_keyframes_at_cuts': True,
        'format': 'best[ext=mp4]'
    }
    if not full_vid:
        ydl_opts['download_ranges'] = download_range_func(None, [(start, end)])
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

json_file = "./short_nlp_conference_list.json"
with open(json_file, "r", encoding="utf-8-sig") as f:
    videos = json.load(f)

# Support both single object and list of objects
if isinstance(videos, dict):
    videos = [videos]

all_talks = []

for idx, video in enumerate(videos, 1):
    url      = video.get("url")
    start    = video.get("start")
    end      = video.get("end")
    full_vid = video.get("full-vid", False)

    # Make output file name unique and human-readable
    title = get_video_title(url)
    safe_title = sanitize_filename(title)
    out_name = f"{safe_title}.mp4"
    print(f"Downloading {url} -> {out_name}")
    download_section(url, out_name, start=start, end=end, full_vid=full_vid)

    print(f"Transcribing {out_name} ...")
    segment_dicts = transcribe_with_whisper(out_name)
    segments = [Segment(start=s["start"], end=s["end"], text=s["text"]) for s in segment_dicts]
    talk = TalkTranscript(video_name=out_name, segments=segments)
    all_talks.append(talk)

save_all_transcripts_to_jsonl(all_talks, "talks_transcripts.jsonl")
print("✅ All transcripts saved to talks_transcripts.jsonl")