"""Policy retrieval: one chunk per handbook section + BM25 ranking.

Pure Python (no embeddings, no downloads) so the logic is fully explainable:
the agent receives only the top-k chunks, never the whole handbook.

Extra signal for "cannot be answered" cases: the result lists query terms that
appear NOWHERE in the handbook (e.g. 'hostel'), so the model can see that a
near-miss chunk (e.g. a refund section) does not actually cover the question.
"""
import math
import re
from collections import Counter
from pathlib import Path

K1, B = 1.5, 0.75
STRONG_SCORE = 4.0  # below this the best chunk is only a weak match

STOP = set(
    """a an the of to in on for and or is are was were be am been do does did i me my we you your
    can could would should will it its this that these those with at by as from if so what how when
    where which who whom about into than then there their them has have had not no yes please tell
    know want need get give make policy rule rules student students handbook let us""".split()
)
SPELLING = {
    "specialization": "specialisation",
    "specializations": "specialisations",
    "program": "programme",
    "programs": "programmes",
    "installment": "instalment",
    "installments": "instalments",
}


def _stem(w: str) -> str:
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _words(text: str):
    """Normalised, stop-word-free tokens (not stemmed)."""
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        t = SPELLING.get(t, t)
        if t in STOP or (len(t) < 2 and not t.isdigit()):
            continue
        out.append(t)
    return out


_HEAD = re.compile(r"^##\s+([A-Z]{3}-\d+)\s*\|\s*(.+?)\s*$", re.M)


def load_chunks(path):
    text = Path(path).read_text(encoding="utf-8")
    heads = list(_HEAD.finditer(text))
    chunks = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        chunks.append({"id": m.group(1), "title": m.group(2), "text": text[m.end():end].strip()})
    return chunks


class PolicyRetriever:
    def __init__(self, handbook_path):
        self.chunks = load_chunks(handbook_path)
        self.total_chars = sum(len(c["text"]) for c in self.chunks)
        # title counted twice so a heading match outweighs a passing mention
        self.tfs = [
            Counter(_stem(w) for w in _words(c["title"] + " " + c["title"] + " " + c["text"]))
            for c in self.chunks
        ]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        self.df = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in self.df.items()}

    def search(self, query: str, k: int = 2):
        words = []
        for w in _words(query):
            if w not in words:
                words.append(w)
        scores = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for w in words:
                f = tf.get(_stem(w), 0)
                if f:
                    s += self.idf[_stem(w)] * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self.avgdl))
            scores.append(s)
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        hits = [
            {**self.chunks[i], "score": round(scores[i], 2)} for i in order if scores[i] > 0
        ]
        top = hits[0]["score"] if hits else 0.0
        return {
            "query": query,
            "chunks": hits,
            "terms_not_in_handbook": [w for w in words if _stem(w) not in self.df],
            "match_quality": "strong" if top >= STRONG_SCORE else "weak",
        }
