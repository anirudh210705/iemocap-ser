"""Parse the official IEMOCAP annotations and create leakage-safe CSV splits."""

from __future__ import annotations

import argparse
import csv
import re
import wave
from collections import Counter
from pathlib import Path


ANNOTATION_RE = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+"
    r"(?P<utterance>\S+)\s+(?P<emotion>\w+)\s+"
    r"\[(?P<valence>[\d.]+),\s*(?P<activation>[\d.]+),\s*(?P<dominance>[\d.]+)\]"
)
TRANSCRIPT_RE = re.compile(r"^(?P<utterance>\S+)\s+\[[^]]+\]:\s*(?P<text>.*)$")
EMOTION_MAP = {"ang": "angry", "hap": "happy", "exc": "happy", "neu": "neutral", "sad": "sad"}
FIELDS = [
    "utterance_id", "audio_path", "session", "speaker", "gender", "method",
    "original_emotion", "label", "start", "end", "duration", "valence",
    "activation", "dominance", "transcript",
]


def load_transcripts(session_dir: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    folder = session_dir / "dialog" / "transcriptions"
    for path in sorted(folder.glob("*.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = TRANSCRIPT_RE.match(line.strip())
            if match:
                transcripts[match["utterance"]] = match["text"].strip()
    return transcripts


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def parse_iemocap(data_root: Path, verify_audio: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for session in range(1, 6):
        session_dir = data_root / f"Session{session}"
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Missing {session_dir}")
        transcripts = load_transcripts(session_dir)
        annotation_dir = session_dir / "dialog" / "EmoEvaluation"
        for annotation in sorted(annotation_dir.glob("*.txt")):
            for line in annotation.read_text(encoding="utf-8", errors="replace").splitlines():
                match = ANNOTATION_RE.match(line.strip())
                if not match or match["emotion"] not in EMOTION_MAP:
                    continue
                utterance = match["utterance"]
                dialogue = utterance.rsplit("_", 1)[0]
                relative_audio = Path(f"Session{session}") / "sentences" / "wav" / dialogue / f"{utterance}.wav"
                audio_path = data_root / relative_audio
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Annotation has no WAV file: {audio_path}")
                gender = utterance.rsplit("_", 1)[1][0]
                duration = wav_duration(audio_path) if verify_audio else float(match["end"]) - float(match["start"])
                rows.append({
                    "utterance_id": utterance,
                    "audio_path": relative_audio.as_posix(),
                    "session": session,
                    "speaker": f"Ses{session:02d}{gender}",
                    "gender": gender,
                    "method": "impro" if "_impro" in utterance else "script",
                    "original_emotion": match["emotion"],
                    "label": EMOTION_MAP[match["emotion"]],
                    "start": float(match["start"]),
                    "end": float(match["end"]),
                    "duration": round(duration, 4),
                    "valence": float(match["valence"]),
                    "activation": float(match["activation"]),
                    "dominance": float(match["dominance"]),
                    "transcript": transcripts.get(utterance, ""),
                })
    if len({row["utterance_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate utterance IDs found")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def create_splits(rows: list[dict[str, object]], output_dir: Path) -> None:
    fixed = output_dir / "fixed"
    write_csv(fixed / "train.csv", [r for r in rows if r["session"] in (1, 2, 3)])
    write_csv(fixed / "val.csv", [r for r in rows if r["session"] == 4])
    write_csv(fixed / "test.csv", [r for r in rows if r["session"] == 5])

    for test_session in range(1, 6):
        val_session = 5 if test_session == 1 else test_session - 1
        fold = output_dir / "loso" / f"fold_{test_session}"
        write_csv(fold / "train.csv", [r for r in rows if r["session"] not in (test_session, val_session)])
        write_csv(fold / "val.csv", [r for r in rows if r["session"] == val_session])
        write_csv(fold / "test.csv", [r for r in rows if r["session"] == test_session])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-audio-verification", action="store_true")
    args = parser.parse_args()

    rows = parse_iemocap(args.data_root, verify_audio=not args.skip_audio_verification)
    write_csv(args.output_dir / "metadata.csv", rows)
    create_splits(rows, args.output_dir / "splits")
    print(f"Prepared {len(rows)} four-class utterances")
    print("Labels:", dict(sorted(Counter(row["label"] for row in rows).items())))
    print("Sessions:", dict(sorted(Counter(row["session"] for row in rows).items())))


if __name__ == "__main__":
    main()

