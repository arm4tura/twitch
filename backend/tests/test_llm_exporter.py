import json

from twitch_cut.llm.exporter import build_notebooklm_package


def _mk_transcript(n_words: int):
    return {
        "segments": [
            {
                "start": 0.0,
                "end": float(n_words),
                "words": [
                    {"word": f"w{i}", "start": float(i), "end": i + 0.5}
                    for i in range(n_words)
                ],
            }
        ]
    }


def test_build_notebooklm_package_writes_all_files(tmp_path):
    m = build_notebooklm_package(_mk_transcript(10), tmp_path, n_highlights=3)
    assert (tmp_path / "prompt.md").exists()
    assert (tmp_path / "schema.json").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "transcript_001.md").exists()
    assert m["n_highlights"] == 3
    assert m["total_words"] == 10


def test_build_notebooklm_package_splits_by_word_limit(tmp_path):
    m = build_notebooklm_package(
        _mk_transcript(10), tmp_path, n_highlights=3, max_words_per_chunk=4
    )
    assert len(m["chunks"]) == 3
    assert (tmp_path / "transcript_003.md").exists()
    body = (tmp_path / "transcript_001.md").read_text(encoding="utf-8")
    assert "chunk 1/3" in body
    assert "[00:00:00.000] w0" in body


def test_build_notebooklm_package_prompt_contains_n(tmp_path):
    build_notebooklm_package(_mk_transcript(5), tmp_path, n_highlights=7)
    prompt = (tmp_path / "prompt.md").read_text(encoding="utf-8")
    assert "7" in prompt
    assert "```json" in prompt


def test_build_notebooklm_package_schema_is_valid_json(tmp_path):
    build_notebooklm_package(_mk_transcript(5), tmp_path, n_highlights=3)
    schema = json.loads((tmp_path / "schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]["highlights"]["items"]["properties"]
    assert set(props) == {"start_s", "end_s", "title", "reason", "score", "quote"}


def test_build_notebooklm_package_rejects_empty_transcript(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        build_notebooklm_package({"segments": []}, tmp_path)


def test_build_notebooklm_package_rejects_too_many_chunks(tmp_path):
    # 51 chunks — превышает NotebookLM лимит 50 источников.
    import pytest

    with pytest.raises(ValueError, match="50 sources"):
        build_notebooklm_package(
            _mk_transcript(51), tmp_path, max_words_per_chunk=1
        )
