# Pelatih Pemula Catur — Chatbot Berbasis TF-IDF

Proyek UAS Kecerdasan Buatan (Kategori D: Chatbot / Virtual Assistant)
Universitas Pamulang • Fakultas Ilmu Komputer

## Deskripsi
Chatbot yang menjawab pertanyaan seputar catur untuk pemula dalam bahasa Indonesia: aturan dasar, gerakan bidak, gerakan khusus, pembukaan, taktik, endgame, notasi, nilai bidak, kesalahan umum, dan cara berlatih.

Chatbot ini **tidak memakai API AI atau LLM**. Pengenalan maksud (intent) dibangun sendiri dengan pemrosesan teks, pembobotan TF-IDF, dan kemiripan kosinus (cosine similarity) terhadap basis pengetahuan di `data/intents.json`.

## Fitur
- Halaman petunjuk dan contoh pertanyaan (sidebar)
- Kolom chat dengan validasi input (kosong, hanya simbol, terlalu panjang)
- Pengenalan intent dengan TF-IDF + cosine similarity
- Fallback untuk pertanyaan di luar topik
- Pertanyaan klarifikasi untuk pesan yang ambigu
- Panel **Lihat proses AI**: hasil praproses, intent terdeteksi, skor, dan 3 kandidat teratas
- Tombol hapus percakapan untuk mengulang skenario uji

## Cara Kerja Singkat
1. **Praproses:** lowercase, hapus tanda baca, tokenisasi, stopword removal, stemming.
2. **Representasi:** tiap contoh kalimat diubah menjadi vektor TF-IDF.
3. **Kemiripan:** pesan pengguna dibandingkan dengan semua contoh kalimat memakai cosine similarity. Skor intent adalah skor tertinggi dari contoh kalimatnya.
4. **Keputusan:**
   - skor tertinggi < ambang τ → fallback
   - selisih skor intent 1 dan 2 < margin δ → tanya balik
   - selain itu → tampilkan jawaban intent terpilih

## Struktur Proyek
```
uas-ai-nim-nama/
├── data/
│   └── intents.json        # basis pengetahuan: intent, contoh kalimat, jawaban
├── src/
│   ├── preprocess.py       # praproses teks
│   ├── tfidf.py            # TF-IDF dan cosine similarity (ditulis sendiri)
│   └── chatbot.py          # threshold, fallback, klarifikasi
├── tests/
│   └── skenario_uji.md     # tabel skenario pengujian
├── app.py                  # antarmuka Streamlit
├── requirements.txt
└── README.md

## Contoh Penggunaan
| Pertanyaan | Hasil yang diharapkan |
|---|---|
| kuda jalannya gimana? | Intent *gerakan bidak* |
| apa itu en passant | Intent *gerakan khusus* |
| cara menang gimana? | Bot menanyakan klarifikasi |
| resep nasi goreng | Fallback (di luar topik) |

## Pengujian
Skenario uji (input, hasil diharapkan, hasil aktual, analisis) ada di `tests/skenario_uji.md` dan pada laporan Bab IV.

## Basis Pengetahuan
`data/intents.json` berisi intent dengan format:
```json
{
  "tag": "gerakan_bidak",
  "label": "gerakan bidak",
  "patterns": ["kuda jalannya gimana", "..."],
  "responses": ["..."]
}
```
Untuk menambah topik, tambahkan intent baru beserta contoh kalimat dan jawabannya, lalu jalankan ulang aplikasi.

## Keterbatasan
- Hanya mencakup catur untuk pemula dan berbahasa Indonesia.
- Basis pengetahuan statis; bot tidak belajar dari percakapan.
- Akurasi bergantung pada keberagaman contoh kalimat di `intents.json`.
- Tidak menganalisis posisi permainan atau memainkan permainan catur.
- Prototipe akademik; jawaban bersifat edukasi umum.

## Sumber dan Referensi
- [Tuliskan sumber materi catur yang dipakai untuk jawaban]
- [Tuliskan referensi TF-IDF dan cosine similarity]
- [Tuliskan pustaka yang dipakai, misalnya Streamlit dan Sastrawi]

## Lisensi
Dibuat untuk keperluan akademik (UAS Kecerdasan Buatan).
