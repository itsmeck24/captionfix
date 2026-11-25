import pymupdf

from captionfix.slides import extract_terms, is_speakable, terms_from_pdf

PAGES = [
    "Cilk Plus and SIMD vectorization\nThe cilk_spawn keyword\nLoop unrolling, unrolling again",
    "Memoization with a hash table. Cilk scheduler and SIMD lanes. See ocw.mit.edu",
]


def test_keeps_rare_words_and_short_acronyms():
    keys = {t.key for t in extract_terms(PAGES)}
    assert {"cilk", "simd", "vectorization", "memoization", "unrolling"} <= keys
    assert "table" not in keys and "with" not in keys and "the" not in keys


def test_skips_code_identifiers_and_urls():
    assert not is_speakable("cilk_spawn")
    assert not is_speakable("ocw.mit.edu")
    assert not is_speakable("12345")
    assert is_speakable("x86")


def test_most_frequent_first_with_written_form():
    terms = extract_terms(PAGES)
    assert terms[0].text == "Cilk" and terms[0].count == 2
    assert "EIE" not in {t.text for t in extract_terms(["EIE once"])}


def test_reads_pdf(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Eigenvalue decomposition and the eigenvector basis")
    doc.save(tmp_path / "deck.pdf")
    keys = {t.key for t in terms_from_pdf(tmp_path / "deck.pdf")}
    assert {"eigenvalue", "eigenvector"} <= keys
