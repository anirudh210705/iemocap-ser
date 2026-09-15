from pathlib import Path

from ser.prepare_iemocap import ANNOTATION_RE, EMOTION_MAP, TRANSCRIPT_RE


def test_annotation_parser() -> None:
    line = "[6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]"
    match = ANNOTATION_RE.match(line)
    assert match is not None
    assert match["utterance"] == "Ses01F_impro01_F000"
    assert EMOTION_MAP[match["emotion"]] == "neutral"


def test_transcript_parser() -> None:
    line = "Ses01F_impro01_F000 [6.2901-8.2357]: Excuse me."
    match = TRANSCRIPT_RE.match(line)
    assert match is not None
    assert match["text"] == "Excuse me."


def test_all_target_labels_are_defined() -> None:
    assert set(EMOTION_MAP.values()) == {"angry", "happy", "neutral", "sad"}

