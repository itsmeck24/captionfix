"""Write site/data.json, the player clip and slide images from results/ and the cached transcripts.

    uv run python scripts/build_site.py

The clip is a short excerpt of an OCW lecture (CC BY-NC-SA), cut with ffmpeg to 48 kbps mono MP3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402

from captionfix import flag  # noqa: E402
from captionfix.captions import Word, clean_reference, read_cues  # noqa: E402
from captionfix.metrics import normalize  # noqa: E402
from captionfix.slides import slide_text, terms_from_pdf  # noqa: E402
from captionfix.sync import SlideIndex  # noqa: E402

SITE = ROOT / "site"
LECTURES = json.loads((ROOT / "data/lectures.json").read_text())["lectures"]
# the player excerpt: a fresh lecture, around a doubt mark that catches a real mistake
# ("immutable or immutable type" where the lecturer said "immutable or mutable")
PLAYER = ("6.0002-05", 696.0)


def load(path: Path) -> list[Word]:
    return [Word.from_dict(d) for d in json.loads(path.read_text())]


def build_player(lec: str, around: float, length: float = 80.0) -> dict:
    """An excerpt for the student player: captions with fixes and doubt marks, the slide that
    was up, and a glossary of the course terms said."""
    runs, src = ROOT / "runs" / lec, ROOT / "data/lectures" / lec
    words = load(runs / "captionfix_zoom_words.json")
    second = load(runs / "plain.json")
    terms = terms_from_pdf(src / "slides.pdf")
    keys = {t.key for t in terms}
    doubt = dict(flag.doubts(words, second, keys))
    course = flag.course_nearby(words, [], keys)
    a = around - length / 3
    a = max(0.0, a - 1.0)
    b = a + length
    name = f"clips/player-{lec.replace('.', '').lower()}.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{a:.2f}", "-to", f"{b:.2f}",
                    "-i", str(src / "audio.flac"), "-ac", "1", "-b:a", "48k", "-map_metadata", "-1",
                    str(SITE / name)], check=True)

    # which slide was up: match each 20 s of the local transcript against the slide text
    pages = slide_text(src / "slides.pdf")
    index = SlideIndex(pages, terms)
    slides, shown = [], {}
    t = a
    while t < b:
        scores = index.match(" ".join(w.text for w in second if t - 10 <= w.start <= t + 30))
        page = max(range(len(pages)), key=lambda p: scores[p])
        if not slides or slides[-1]["page"] != page:
            slides.append({"t": round(t - a, 2), "page": page})
            shown[page] = None
        t += 20.0
    import pymupdf

    with pymupdf.open(src / "slides.pdf") as doc:
        for page in shown:
            out = SITE / f"media/{lec.replace('.', '').lower()}-p{page}.png"
            out.parent.mkdir(exist_ok=True)
            doc[page].get_pixmap(dpi=60).save(out)
    for sl in slides:
        sl["src"] = f"media/{lec.replace('.', '').lower()}-p{sl['page']}.png"

    out_words, glossary = [], {}
    for i, w in enumerate(words):
        if not a <= w.start < b:
            continue
        kind = "fixed" if "corrected" in w.flags else "doubt" if i in doubt else None
        out_words.append({"t": round(w.start - a, 2), "e": round(w.end - a, 2), "text": w.text, "kind": kind,
                          "was": w.original if kind == "fixed" else (doubt.get(i) or "").rstrip(".,?!;:")})
        if course[i]:
            key = w.text.strip(".,?!;:").lower()
            glossary.setdefault(key, {"term": w.text.strip(".,?!;:"), "t": round(w.start - a, 2)})
    meta = next(m for m in LECTURES if m["id"] == lec)
    return {"lecture": lec, "course": meta["course"], "title": meta["title"], "audio": name,
            "start": round(a, 2), "words": out_words, "slides": slides,
            "glossary": sorted(glossary.values(), key=lambda g: g["t"])}


# Hand-checked against the audio: Zoom wrote something else, and MIT's captions have the term
# right. Most automatic candidates were spacing ("pop count"), plurals, or typos in MIT's captions.
QUIZ = [("6.172-04", "unaligned", "on aligned"), ("6.172-09", "inlining", "inline"),
        ("6.172-03", "XOR", "X or")]


def misheard(picks: list[tuple[str, str, str]]) -> list[dict]:
    """The caption line around each picked term as Zoom wrote it, and a short audio clip."""
    out = []
    for k, (lec, term, zoom_text) in enumerate(picks):
        src = ROOT / "data/lectures" / lec
        ref = normalize(" ".join(clean_reference(c.text) for c in read_cues(src / "reference.srt")))
        words = load(ROOT / "runs" / lec / "zoom.json")
        hyp, owner = [], []
        for i, w in enumerate(words):
            for t in normalize(w.text):
                hyp.append(t)
                owner.append(i)
        o = jiwer.process_words(" ".join(ref), " ".join(hyp))
        for c in o.alignments[0]:
            r, h = ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]
            if c.type != "substitute" or r != [term.lower()]:
                continue
            # the alignment can split the miss ("risk" inserted, "prediction" swapped): find Zoom's words
            def bare(text: str) -> str:
                return text.strip(".,?!;:").lower()

            want = [bare(t) for t in zoom_text.split()]
            near = owner[c.hyp_start_idx]
            i = next(i for i in range(near - 3, near + 3)
                     if [bare(w.text) for w in words[i:i + len(want)]] == want)
            j = i + len(want) - 1
            before = " ".join(w.text for w in words[max(0, i - 7):i])
            after = " ".join(w.text for w in words[j + 1:j + 7])
            wrong = " ".join(w.text for w in words[i:j + 1])
            t0, t1 = words[max(0, i - 7)].start, words[min(len(words) - 1, j + 6)].end
            name = f"clips/quiz-{k}.mp3"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0 - 0.3:.2f}", "-to", f"{t1 + 0.3:.2f}",
                            "-i", str(src / "audio.flac"), "-ac", "1", "-b:a", "48k", "-map_metadata", "-1",
                            str(SITE / name)], check=True)
            meta = next(m for m in LECTURES if m["id"] == lec)
            out.append({"said": term, "zoom": " ".join(h), "wrong": wrong.strip(".,?!"), "before": before,
                        "after": after, "lecture": lec, "course": meta["course"].split(" (")[0],
                        "at": f"{int(words[i].start // 60)}:{int(words[i].start % 60):02d}", "audio": name})
            break
    return out


def main() -> None:
    test = json.loads((ROOT / "results/test.json").read_text())
    fresh_path = ROOT / "results/fresh.json"
    fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else None
    dev = json.loads((ROOT / "results/dev_tuning_zoom.json").read_text())
    (SITE / "clips").mkdir(parents=True, exist_ok=True)
    for old in list((SITE / "clips").glob("*.mp3")) + list((SITE / "media").glob("*-p*.png")):
        old.unlink()

    def outcome(name: str, res: dict) -> dict:
        z, cf, r = res["systems"]["zoom"], res["systems"]["captionfix_zoom"], res["review_zoom"]
        return {"name": name, "terms": z["course_total"], "zoom": z["course_mistakes"],
                "captionfix": cf["course_mistakes"], "caught": r["course_first@0.1"]["course_share_caught"],
                "caught_all": r["course_first@0.1"]["share_caught"],
                "random": r["random@0.1"]["course_share_caught"],
                "minutes": r["course_first@0.1"]["review_minutes"] / (res["minutes"] / 60)}

    rows = [outcome(f"Test: {test['lectures']} lectures, 3 courses", test)]
    if fresh:
        rows.append(outcome(f"Fresh: {fresh['lectures']} lectures, a course never looked at", fresh))
    z = test["systems"]["zoom"]
    main_row = rows[-1] if fresh else rows[0]
    per = z["course_total"] / max(1, z["course_mistakes"])
    data = {
        "test": test, "fresh": fresh, "vocab": dev["vocabulary"],
        "outcome": rows,
        "stats": [
            {"value": f"1 in {per:.0f}", "unit": "course terms",
             "label": f"Zoom gets wrong, or writes when it was not said, on {test['minutes'] / 60:.0f} hours of MIT lectures"},
            {"value": "\u2013".join(f"{100 * r['caught']:.0f}" for r in rows) + "%", "unit": "of them",
             "label": "land in the 10% of the lecture CaptionFix sends a TA to, on the test and fresh lectures "
                      f"({100 * main_row['random']:.0f}% for random spot checks)"},
            {"value": f"{main_row['minutes']:.0f}", "unit": "minutes",
             "label": "of TA time per lecture hour, instead of rewatching the whole hour"},
            {"value": f"{test['lectures'] + (fresh['lectures'] if fresh else 0)}", "unit": "lectures",
             "label": f"from {4 if fresh else 3} MIT courses, scored against MIT's human captions"},
        ],
        "problem": (f"On {test['lectures']} MIT test lectures Zoom's captions were {100 * (1 - z['wer']):.0f}% right "
                    f"word for word, yet about 1 in {per:.0f} course terms came out wrong."),
        "intro": (f"{test['lectures']} test lectures ({test['minutes'] / 60:.1f} hours) from three MIT courses"
                  + (f", plus {fresh['lectures']} lectures ({fresh['minutes'] / 60:.1f} hours) from a fourth course that "
                     "was only downloaded after every setting was fixed" if fresh else "")
                  + ". Every setting was chosen on 3 separate tuning lectures by rules written down before scoring. "
                  "A course term is a word from the slides or a rare word (Zipf frequency under 3). A course-term "
                  "mistake is one the captions got wrong, or one the transcript wrote that MIT's captions do not have."),
        "misheard": misheard(QUIZ),
        "player": build_player(*PLAYER),
        "notes": json.loads((ROOT / "results/notes.json").read_text()),
    }
    (SITE / "data.json").write_text(json.dumps(data, indent=1))
    print("wrote site/data.json, player", data["player"]["lecture"], data["player"]["start"])


if __name__ == "__main__":
    main()
