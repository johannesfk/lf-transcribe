from sermon_transcribe.prose import paragraphs_from_segments
from sermon_transcribe.asr import Segment


def test_paragraph_grouping_basic():
    segs = [
        Segment(0.0, 5.0, "Now we begin. This is the first point."),
        Segment(7.6, 12.5, "So consider this carefully. It matters."),
        Segment(15.2, 20.0, "Finally, we conclude. Amen."),
    ]
    paras = paragraphs_from_segments(segs, paragraph_gap_s=2.0, max_sentences=4)
    assert len(paras) >= 2
    all_text = "\n".join(p.text for p in paras)
    assert "Amen" in all_text
