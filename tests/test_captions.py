from captionfix.captions import (Word, clean_reference, cues_to_words, fmt_time, parse_time, read_cues,
                                 words_to_cues, write_vtt)

ZOOM_VTT = """WEBVTT

1
00:00:01.200 --> 00:00:04.000
Jane Smith: Today we look at the Cilk runtime.

2
00:00:04.500 --> 00:00:06.000
Jane Smith: Any questions?
"""

OCW_SRT = """1
00:00:00,790 --> 00:00:03,130
PROFESSOR: So let's start. [LAUGHTER]

2
00:01:02,500 --> 00:01:05,000
AUDIENCE: What about SIMD?
"""


def test_times_round_trip():
    assert parse_time("01:02:03.450") == 3723.45
    assert parse_time("00:01:02,500") == 62.5
    assert parse_time("02:03.450") == 123.45
    assert fmt_time(3723.45) == "01:02:03.450"


def test_reads_zoom_transcript_without_speaker_names(tmp_path):
    p = tmp_path / "zoom.vtt"
    p.write_text(ZOOM_VTT)
    cues = read_cues(p)
    assert [c.start for c in cues] == [1.2, 4.5]
    words = cues_to_words(cues)
    assert [w.text for w in words][:3] == ["Today", "we", "look"]
    assert words[0].start == 1.2 and abs(words[6].end - 4.0) < 1e-9


def test_reference_drops_labels_and_sound_notes(tmp_path):
    p = tmp_path / "ref.srt"
    p.write_text(OCW_SRT)
    text = " ".join(clean_reference(c.text) for c in read_cues(p))
    assert text.split() == ["So", "let's", "start.", "What", "about", "SIMD?"]
    assert "PROFESSOR" not in text and "AUDIENCE" not in text and "LAUGHTER" not in text


def test_sentence_start_is_not_a_speaker_label():
    assert clean_reference("Note: this is kept") == "Note: this is kept"


def test_cues_respect_two_lines_of_42_characters():
    words = [Word(f"word{i:02d}", i * 0.3, i * 0.3 + 0.25) for i in range(60)]
    cues = words_to_cues(words)
    for c in cues:
        lines = c.text.split("\n")
        assert len(lines) <= 2
        assert all(len(line) <= 42 for line in lines)
        assert c.end - c.start <= 6.0 + 1e-9
    assert " ".join(c.text.replace("\n", " ") for c in cues).split() == [w.text for w in words]


def test_cues_break_at_long_pauses_and_stay_up_one_second():
    words = [Word("Hi.", 0.0, 0.2), Word("Later", 5.0, 5.4)]
    cues = words_to_cues(words)
    assert len(cues) == 2
    assert cues[0].end == 1.0


def test_write_vtt(tmp_path):
    p = tmp_path / "out.vtt"
    write_vtt(words_to_cues([Word("Hello", 0, 0.5), Word("there.", 0.5, 1.2)]), p)
    assert p.read_text().startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.200\nHello there.")
