"""Evaluasi: `python -m tests.evaluate tests/data/uji_baru.json [--detail]`.

Metrik:
  dalam domain  = intent terprediksi sama persis dengan harapan
  luar domain   = jatuh ke fallback
  ambigu        = tepat: bot bertanya balik; aman: bertanya balik ATAU fallback
                  (tidak menjawab dengan yakin)
"""
import json
import sys
from pathlib import Path

from src.nlp_engine import ChessChatbotAI

ROOT = Path(__file__).resolve().parent.parent


def load_bot(**kw):
    return ChessChatbotAI(str(ROOT / "data" / "knowledge_base.json"), **kw)


def evaluate(bot, cases):
    stat = {"in": [0, 0], "ood": [0, 0], "amb_tepat": [0, 0], "amb_aman": [0, 0]}
    gagal = []
    for c in cases:
        r = bot.predict(c["input"])
        exp = c["expected"]
        if exp == "fallback":
            ok = r["status"] == "fallback"
            stat["ood"][0] += ok; stat["ood"][1] += 1
        elif exp == "klarifikasi":
            stat["amb_tepat"][1] += 1; stat["amb_aman"][1] += 1
            stat["amb_tepat"][0] += r["status"] == "klarifikasi"
            ok = r["status"] in ("klarifikasi", "fallback")
            stat["amb_aman"][0] += ok
        else:
            ok = r["status"] == "dijawab" and r["predicted_intent"] == exp
            stat["in"][0] += ok; stat["in"][1] += 1
        if not ok:
            gagal.append((c["input"], exp, r["status"], r["predicted_intent"], r["top_candidates"][:2], r["tokens"], r["coverage"]))
    return stat, gagal


def ringkas(stat):
    return " | ".join(f"{k}: {a}/{b}" for k, (a, b) in stat.items())


def tulis_markdown(bot, cases, path):
    """Tabel skenario uji (input, harapan, aktual, analisis) untuk laporan Bab IV."""
    baris = ["| No | Input | Harapan | Aktual | Skor | Sesuai | Analisis |", "|---|---|---|---|---|---|---|"]
    for n, c in enumerate(cases, 1):
        r = bot.predict(c["input"])
        exp = c["expected"]
        aktual = r["predicted_intent"] if r["status"] == "dijawab" else r["status"]
        sesuai = aktual == exp
        baris.append(f"| {n} | {c['input']} | {exp} | {aktual} | {r['confidence']:.2f} | "
                     f"{'ya' if sesuai else 'TIDAK'} | {r['reasoning']} |")
    Path(path).write_text("\n".join(baris) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1], encoding="utf-8"))
    stat, gagal = evaluate(load_bot(), cases)
    print(ringkas(stat))
    if "--md" in sys.argv:
        out = ROOT / "tests" / ("hasil_" + Path(sys.argv[1]).stem + ".md")
        tulis_markdown(load_bot(), cases, out)
        print("tabel ditulis ke", out)
    if "--detail" in sys.argv:
        for g in gagal:
            print("GAGAL:", g)
