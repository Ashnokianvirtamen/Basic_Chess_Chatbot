# ChessMate AI — Asisten Catur Pemula Berbasis TF-IDF

Proyek UAS Kecerdasan Buatan (Kategori D: Chatbot / Virtual Assistant)
Universitas Pamulang • Fakultas Ilmu Komputer

| | |
|---|---|
| Nama | [Nama Lengkap] |
| NIM | [NIM] |
| Kelas | [Kelas] |
| Dosen pengampu | [Nama Dosen] |
| Tahun akademik | [Semester / Tahun Akademik] |

## Deskripsi
Chatbot berbahasa Indonesia yang menjawab pertanyaan catur untuk pemula. Chatbot ini **tidak memakai API AI atau LLM**. Maksud (intent) pertanyaan dikenali dengan praproses teks, pembobotan **TF-IDF** (scikit-learn, unigram + bigram), dan **cosine similarity** terhadap basis pengetahuan `data/knowledge_base.json` (22 intent, ±240 contoh kalimat).

## Fitur
- Halaman petunjuk, topik, dan contoh pertanyaan (sidebar)
- Kolom chat dengan validasi input (kosong, hanya simbol, lebih dari 200 karakter)
- Pengenalan intent TF-IDF + cosine similarity
- Normalisasi bahasa gaul dan sinonim (mis. *ratu* → menteri, *gmn* → bagaimana)
- Koreksi typo berbasis kemiripan string (mis. *en pasant* → *en passant*)
- Fallback untuk pertanyaan di luar topik
- Pertanyaan klarifikasi (jawaban ditahan) saat dua intent hampir sama skornya
- Panel **Lihat proses AI**: hasil praproses, koreksi typo, cakupan kosakata, 3 intent teratas, alasan keputusan
- Tombol hapus percakapan

## Cara Kerja
1. **Validasi input.**
2. **Praproses:** lowercase → hapus tanda baca → tokenisasi → normalisasi notasi/sinonim/gaul → stopword → stemming Sastrawi (istilah catur dilindungi dari stemming) → sapaan dibuang bila ada pertanyaan inti.
3. **Koreksi typo:** kata tak dikenal diganti kata kosakata yang sangat mirip (`difflib`, batas 0,88).
4. **TF-IDF + cosine similarity** terhadap semua contoh kalimat; skor intent = skor tertinggi dari contoh kalimatnya.
5. **Keputusan:**
   - skor tertinggi < τ **atau** cakupan kosakata < minimum → **fallback**
   - selisih skor intent 1 dan 2 < δ → **klarifikasi**
   - selain itu → **jawab**

Parameter: τ = 0,22; δ = 0,08; cakupan minimum = 0,5; batas typo = 0,88. Nilai δ, cakupan, dan batas typo dipilih lewat grid search pada set **dev** (`tests/data/dev.json`); ambang τ dipertahankan di nilai awal karena skor dev sama di rentang 0,15–0,30.

## Struktur Proyek
```
chessmate-ai/
├── data/knowledge_base.json     # 22 intent: pola + jawaban + fallback
├── src/nlp_engine.py            # mesin NLP
├── tests/
│   ├── data/dev.json            # set tuning parameter
│   ├── data/uji_baru.json       # set uji, tidak dipakai untuk tuning
│   ├── data/benchmark_lama.json # benchmark awal (kegagalannya sudah pernah dilihat)
│   └── evaluate.py              # metrik + tabel skenario uji
├── app.py                       # antarmuka Gradio
├── requirements.txt
└── README.md
```

## Instalasi dan Menjalankan
```bash
git clone https://github.com/<username>/uas-ai-nim-nama.git
cd uas-ai-nim-nama
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Pengujian
```bash
python -m tests.evaluate tests/data/uji_baru.json --detail --md
```
`--md` menulis tabel skenario uji (input, harapan, aktual, analisis) ke `tests/hasil_uji_baru.md` untuk laporan Bab IV.

### Hasil evaluasi
| Set | Dalam domain (intent tepat) | Luar domain (fallback) | Ambigu (klarifikasi/fallback) |
|---|---|---|---|
| Dev (tuning) | 40/44 | 7/8 | 3/5 |
| **Uji baru** | **43/46** | **7/10** | **5/8** |
| Benchmark lama* | 32/33 | 7/8 | 3/5 |

\*Kegagalan benchmark lama sudah dilihat sebelum perbaikan, sehingga tidak lagi murni.

Sampel kecil, ditulis oleh pengembang; angka ini indikatif, bukan estimasi akurasi umum.

## Keterbatasan
- **Luar domain:** pertanyaan yang memuat satu kata domain (mis. *kuda lumping*, *cara main futsal*) bisa lolos karena cakupan kosakata hanya diukur kasar.
- **Ambigu:** pertanyaan sangat pendek (mis. *cara menang gimana*) bisa dijawab satu intent bila selisih skornya tipis di atas δ.
- **Pola bertabrakan:** *bot ini bisa apa aja* dijawab sapaan karena pola sapaan berisi kata "bot".
- Pertanyaan berisi dua topik sekaligus hanya dijawab satu intent.
- Basis pengetahuan statis, bahasa Indonesia saja, hanya catur untuk pemula.
- Akurasi bergantung pada keberagaman contoh kalimat di basis pengetahuan.

## Sumber dan Referensi
- [Sumber materi catur untuk jawaban]
- [Referensi TF-IDF, cosine similarity, stemming Nazief-Adriani/Sastrawi]
- scikit-learn, Sastrawi, Gradio

## Pernyataan Penggunaan AI Generatif
[Isi dengan jujur. Contoh: Claude (Anthropic) digunakan untuk membantu menyusun kerangka rancangan, menggabungkan dan memperluas basis pengetahuan, menulis modul normalisasi/klarifikasi, dan README. Seluruh kode telah ditinjau dan dapat dijelaskan oleh penulis.]
