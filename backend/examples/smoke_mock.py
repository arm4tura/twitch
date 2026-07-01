from __future__ import annotations

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from twitch_cut.config import PipelineConfig
from twitch_cut.decisions import build_decisions, write_decisions
from twitch_cut.profanity import RussianNormalizer, detect_profanity, load_banwords
from twitch_cut.timecode import parse_timecode
from twitch_cut.transcription import load_mock_transcript
from twitch_cut.vegas_export import write_vegas_script


def main() -> None:
    """Run a dependency-light smoke test without ffmpeg, WhisperX, or Typer."""

    normalizer = RussianNormalizer()
    transcript = load_mock_transcript(BACKEND_ROOT / "examples" / "mock_transcript.json")
    banwords = load_banwords(BACKEND_ROOT / "banwords.example.txt", normalizer=normalizer)

    range_in = parse_timecode("00:01:00")
    range_out = parse_timecode("00:02:00")
    matches = detect_profanity(transcript, banwords, range_in, normalizer=normalizer)

    decisions = build_decisions(
        stream_path=PROJECT_ROOT / "input" / "stream.mp4",
        original_path=PROJECT_ROOT / "input" / "original_video.mp4",
        range_in=range_in,
        range_out=range_out,
        matches=matches,
        config=PipelineConfig(),
        transcript_cache=BACKEND_ROOT / "examples" / "mock_transcript.json",
        audio_cache=None,
    )

    output_dir = PROJECT_ROOT / "output"
    write_decisions(output_dir / "smoke_decisions.json", decisions)
    write_vegas_script(output_dir / "smoke_vegas_build.cs", decisions)

    print("Smoke test OK")
    print(f"Mute markers: {len(matches)}")
    print(f"decisions: {output_dir / 'smoke_decisions.json'}")
    print(f"vegas:     {output_dir / 'smoke_vegas_build.cs'}")


if __name__ == "__main__":
    main()
