"""Mesin NLP ChessMate AI.

Inti (TF-IDF + cosine similarity dengan scikit-learn, agregasi skor per intent,
fallback berbasis ambang) berasal dari rancangan awal penulis. Ditambahkan:
  * normalisasi singkatan/sinonim istilah catur
  * koreksi typo berbasis kemiripan string terhadap kosakata
  * pembuangan sapaan bila ada pertanyaan inti
  * pengukuran cakupan kosakata untuk menolak pertanyaan di luar domain
  * pertanyaan klarifikasi saat dua intent hampir sama skornya
  * validasi input

Alur: validasi -> praproses -> koreksi typo -> TF-IDF -> cosine similarity
      -> skor per intent -> keputusan (fallback / klarifikasi / jawab).
"""
import difflib
import json
import random
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ---------------------------------------------------------------- parameter
MAX_LEN = 200
CONFIDENCE_THRESHOLD = 0.22   # skor minimum; di bawahnya -> fallback
AMBIGUITY_MARGIN = 0.08       # selisih skor 1 vs 2; di bawahnya -> klarifikasi
MIN_COVERAGE = 0.5            # porsi kata yang dikenal kosakata; di bawahnya -> fallback
TYPO_CUTOFF = 0.88            # kemiripan minimum untuk koreksi typo

# ---------------------------------------------------------------- kamus
# Singkatan, bahasa gaul, dan sinonim -> bentuk yang dipakai basis pengetahuan.
SINONIM = {
    # istilah catur
    "ratu": "menteri", "permaisuri": "menteri", "queen": "menteri",
    "prajurit": "pion", "pawn": "pion", "knight": "kuda", "bishop": "gajah",
    "peluncur": "gajah", "rook": "benteng", "king": "raja",
    "castling": "rokade", "checkmate": "skakmat", "check": "skak",
    "seri": "remis", "draw": "remis", "opening": "pembukaan",
    "pasant": "passant", "pasan": "passant",
    "miring": "diagonal", "serong": "diagonal", "kotak": "petak",
    "loncat": "lompat", "meloncat": "lompat", "lompatin": "lompat",
    "makan": "pukul", "dimakan": "pukul", "memakan": "pukul", "diambil": "pukul",
    "jago": "mahir", "gerakannya": "gerak", "gerakan": "gerak", "geraknya": "gerak",
    "bergerak": "gerak", "jalan": "gerak", "jalannya": "gerak", "melangkah": "gerak",
    # bahasa gaul / singkatan umum
    "gmn": "bagaimana", "gimana": "bagaimana", "gmna": "bagaimana", "bgmn": "bagaimana",
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak", "tdk": "tidak",
    "udah": "sudah", "udh": "sudah", "sdh": "sudah", "aja": "saja", "yg": "yang",
    "dgn": "dengan", "utk": "untuk", "buat": "untuk", "jg": "juga", "krn": "karena",
    "kalo": "kalau", "klo": "kalau", "knp": "mengapa", "kenapa": "mengapa", "kok": "mengapa",
    "makasih": "terima kasih", "thx": "terima kasih", "thanks": "terima kasih",
    "tq": "terima kasih", "trims": "terima kasih",
    "hi": "halo", "hello": "halo", "hai": "halo", "p": "halo", "sore": "siang",
    "sampe": "sampai", "bye": "pamit", "dadah": "pamit",
}

STOPWORDS = {
    "yang", "untuk", "pada", "ke", "di", "dari", "ini", "itu", "dengan", "dan", "atau",
    "adalah", "yaitu", "secara", "bisa", "dapat", "apakah", "bagaimana", "cara", "apa",
    "kapan", "siapa", "mengapa", "berapa", "saja", "ada", "saat", "jika", "bila", "kalau",
    "tersebut", "kita", "dong", "sih", "nih", "ya", "yah", "kah", "nya", "deh", "tuh",
    "kan", "lah", "mana", "kemana", "boleh", "tidak", "sudah", "mau", "aku", "saya", "kamu", "bagus",
}

# Kata sapaan: dibuang bila masih ada kata lain (agar "halo, <pertanyaan>" dijawab pertanyaannya).
SAPAAN = {"halo", "selamat", "pagi", "siang", "malam", "assalamualaikum", "permisi",
          "bot", "chess", "chessmate", "coach", "kak"}

# Istilah domain yang tidak di-stem (Sastrawi bisa memotongnya salah, mis. "menteri" -> "teri").
PROTECTED = {"menteri", "pion", "rokade", "skakmat", "skak", "stalemate", "endgame", "gajah",
             "benteng", "kuda", "raja", "passant", "remis", "notasi", "taktik", "puzzle",
             "petak", "bidak", "promosi", "fork", "pin", "skewer", "rating", "diagonal", "gerak"}

# Notasi langkah, mis. e4, Nf3, Bxc6 -> satu token "notasi".
_POLA_NOTASI = re.compile(r"^[kqrbn]?[a-h]?x?[a-h][1-8]$")

_stemmer = StemmerFactory().create_stemmer()


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    if word in PROTECTED:
        return word
    return _stemmer.stem(word)


def preprocess_tokens(text: str) -> List[str]:
    """Case folding -> hapus tanda baca -> tokenisasi -> notasi/sinonim -> stopword
    -> stemming -> buang sapaan bila ada kata lain."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens: List[str] = []
    for tok in text.split():
        if _POLA_NOTASI.match(tok):
            tok = "notasi"
        tokens.extend(SINONIM.get(tok, tok).split())
    kept = [t for t in tokens if t not in STOPWORDS]
    tokens = kept if kept else tokens          # jika semua stopword, pakai apa adanya
    tokens = [_stem(t) for t in tokens if len(t) > 1]
    inti = [t for t in tokens if t not in SAPAAN]
    return inti if inti else tokens


def preprocess_text(text: str) -> str:
    return " ".join(preprocess_tokens(text))


def validate_input(text: Optional[str]) -> Optional[str]:
    """Pesan kesalahan, atau None bila input valid."""
    if text is None or not text.strip():
        return "Pesan tidak boleh kosong."
    if len(text) > MAX_LEN:
        return f"Pesan terlalu panjang (maksimal {MAX_LEN} karakter)."
    if not re.search(r"[A-Za-z0-9]", text):
        return "Pesan harus berisi huruf atau angka, bukan simbol saja."
    return None


class ChessChatbotAI:
    def __init__(self, kb_path: str = "data/knowledge_base.json",
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 ambiguity_margin: float = AMBIGUITY_MARGIN,
                 min_coverage: float = MIN_COVERAGE,
                 typo_cutoff: float = TYPO_CUTOFF):
        self.typo_cutoff = typo_cutoff
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.min_coverage = min_coverage
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.intents: List[Dict[str, Any]] = data["intents"]
        self.fallback_data: Dict[str, Any] = data.get("fallback", {})
        self.labels = {i["tag"]: i["tag"].replace("_", " ") for i in self.intents}
        self.train_model()

    # ------------------------------------------------------------ pelatihan
    def train_model(self) -> None:
        """Praproses semua pola lalu bentuk matriks TF-IDF (unigram + bigram)."""
        self.pattern_intent_map: List[str] = []
        texts: List[str] = []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                self.pattern_intent_map.append(intent["tag"])
                texts.append(preprocess_text(pattern))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.vocab = {w for w in self.vectorizer.vocabulary_ if " " not in w}

    # ------------------------------------------------------------ typo
    def correct_typos(self, tokens: List[str]):
        """Ganti kata yang tak dikenal dengan kata kosakata yang sangat mirip."""
        out, fixes = [], []
        for t in tokens:
            if t not in self.vocab and len(t) >= 4:
                m = difflib.get_close_matches(t, self.vocab, n=1, cutoff=self.typo_cutoff)
                if m:
                    fixes.append((t, m[0]))
                    t = m[0]
            out.append(t)
        return out, fixes

    # ------------------------------------------------------------ prediksi
    def _hasil(self, status: str, response: str, **extra) -> Dict[str, Any]:
        base = {"status": status, "response": response, "predicted_intent": extra.pop("intent", None),
                "confidence": 0.0, "is_fallback": status == "fallback",
                "is_ambiguous": status == "klarifikasi", "tokens": [], "corrections": [],
                "coverage": 0.0, "top_candidates": [], "reasoning": ""}
        base.update(extra)
        return base

    def predict(self, user_input: str) -> Dict[str, Any]:
        err = validate_input(user_input)
        if err:
            return self._hasil("error", err, reasoning="Input tidak lolos validasi.")

        tokens = preprocess_tokens(user_input)
        tokens, fixes = self.correct_typos(tokens)
        known = [t for t in tokens if t in self.vocab]
        coverage = len(known) / len(tokens) if tokens else 0.0

        sims = cosine_similarity(self.vectorizer.transform([" ".join(tokens)]), self.tfidf_matrix).flatten()
        scores: Dict[str, float] = {}
        for tag, s in zip(self.pattern_intent_map, sims):
            if s > scores.get(tag, 0.0):
                scores[tag] = float(s)
        ranking = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        top = [{"intent": t, "score": round(s, 4)} for t, s in ranking[:3]]
        best_tag, best = ranking[0] if ranking else (None, 0.0)
        info = dict(tokens=tokens, corrections=fixes, coverage=round(coverage, 2),
                    top_candidates=top, confidence=round(best, 4))

        fb = self.fallback_data.get("default_responses", ["Maaf, saya belum paham pertanyaanmu."])
        if best < self.confidence_threshold:
            return self._hasil("fallback", random.choice(fb), **info, reasoning=(
                f"Skor tertinggi {best:.2f} di bawah ambang {self.confidence_threshold:.2f}."))
        if coverage < self.min_coverage:
            return self._hasil("fallback", random.choice(fb), **info, reasoning=(
                f"Hanya {coverage:.0%} kata dikenal basis pengetahuan (minimum {self.min_coverage:.0%}); "
                "kemungkinan di luar domain catur."))

        second_tag, second = ranking[1] if len(ranking) > 1 else (None, 0.0)
        if second_tag and second >= self.confidence_threshold and (best - second) < self.ambiguity_margin:
            msg = (f"Pertanyaanmu agak ambigu. Maksudmu **{self.labels[best_tag]}** "
                   f"atau **{self.labels[second_tag]}**? Coba tulis lebih spesifik.")
            return self._hasil("klarifikasi", msg, **info, reasoning=(
                f"Selisih skor {best:.2f} vs {second:.2f} kurang dari margin {self.ambiguity_margin:.2f}."))

        intent = next(i for i in self.intents if i["tag"] == best_tag)
        return self._hasil("dijawab", random.choice(intent["responses"]), intent=best_tag, **info,
                           reasoning=f"Intent '{best_tag}' terpilih dengan cosine similarity {best:.2f}.")
