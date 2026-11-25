"""Antarmuka Gradio ChessMate AI dengan desain modern ala dashboard AI.
Jalankan dengan: python app.py
"""
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from src.nlp_engine import MAX_LEN, ChessChatbotAI

ASSETS_DIR = Path(__file__).parent / "assets"


def get_image_data_uri(path: Path) -> str:
    """Konversi file gambar lokal ke Base64 data URI agar selalu terload di browser."""
    if not path.exists():
        return ""
    ext = path.suffix.lower().lstrip(".")
    mime_map = {
        "svg": "image/svg+xml",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    mime = mime_map.get(ext, f"image/{ext}")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


LOGO_URI = get_image_data_uri(ASSETS_DIR / "chesslogo.svg")
GUIDEBOOK_URI = get_image_data_uri(ASSETS_DIR / "guidebook.png")

# Daftar 6 contoh pertanyaan catur dalam bentuk kartu dengan gambar asli dari assets/
KARTU_CONTOH = [
    {
        "icon": str(ASSETS_DIR / "knight.png"),
        "title": "Gerakan Kuda",
        "desc": "Kuda jalannya gimana?",
        "query": "Kuda jalannya gimana?",
    },
    {
        "icon": str(ASSETS_DIR / "enpassant.webp"),
        "title": "Aturan En Passant",
        "desc": "Apa itu en passant?",
        "query": "Apa itu en passant?",
    },
    {
        "icon": str(ASSETS_DIR / "castle.webp"),
        "title": "Syarat Rokade",
        "desc": "Syarat rokade apa saja?",
        "query": "Syarat rokade apa saja?",
    },
    {
        "icon": str(ASSETS_DIR / "checkmate.jpg"),
        "title": "Skak vs Skakmat",
        "desc": "Apa beda skak dan skakmat?",
        "query": "Apa beda skak dan skakmat?",
    },
    {
        "icon": str(ASSETS_DIR / "chessboard.png"),
        "title": "Notasi Catur",
        "desc": "Arti Nf3 itu apa?",
        "query": "Arti Nf3 itu apa?",
    },
    {
        "icon": str(ASSETS_DIR / "rook.png"),
        "title": "Teknik Endgame",
        "desc": "Cara skakmat pakai raja & benteng",
        "query": "Cara skakmat pakai raja dan benteng",
    },
]

STATUS = {
    "dijawab": "[Dijawab]",
    "klarifikasi": "[Perlu Klarifikasi]",
    "fallback": "[Fallback]",
    "error": "[Input Tidak Valid]",
}


def load_bot() -> ChessChatbotAI:
    return ChessChatbotAI(str(Path(__file__).parent / "data" / "knowledge_base.json"))


bot = load_bot()


def format_detail_markdown(res: Dict[str, Any]) -> str:
    status_label = STATUS.get(res.get("status", ""), res.get("status", ""))
    tokens_str = f"`{res.get('tokens', [])}`"

    corrections_line = ""
    if res.get("corrections"):
        corrections_line = "\n- **Koreksi typo:** " + ", ".join(
            f"`{a}` -> `{b}`" for a, b in res["corrections"]
        )

    coverage_val = res.get("coverage", 0.0)
    coverage_str = f"{coverage_val:.0%}"

    top_candidates_md = ""
    if res.get("top_candidates"):
        rows = [f"| `{c['intent']}` | {c['score']:.4f} |" for c in res["top_candidates"]]
        top_candidates_md = (
            "\n\n**3 intent teratas:**\n\n"
            "| Intent | Skor kemiripan |\n"
            "| :--- | :--- |\n"
            + "\n".join(rows)
        )

    reasoning_text = res.get("reasoning", "")
    params_text = (
        f"Ambang tau = {bot.confidence_threshold}, "
        f"margin delta = {bot.ambiguity_margin}, "
        f"cakupan minimum = {bot.min_coverage}"
    )

    return (
        f"\n\n<details class='ai-process-details'><summary><b>Lihat proses AI</b></summary>\n\n"
        f"- **Status:** {status_label}\n"
        f"- **Hasil praproses:** {tokens_str}"
        f"{corrections_line}\n"
        f"- **Cakupan kosakata:** {coverage_str}"
        f"{top_candidates_md}\n\n"
        f"*{reasoning_text}*\n\n"
        f"<small><i>{params_text}</i></small>\n"
        f"</details>"
    )


def chat_response(user_message: str, history: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not user_message or not user_message.strip():
        return "", history or []

    res = bot.predict(user_message)
    content = res["response"]
    if res.get("status") != "error":
        content += format_detail_markdown(res)

    updated_history = list(history) if history else []
    updated_history.append({"role": "user", "content": user_message})
    updated_history.append({"role": "assistant", "content": content})

    return "", updated_history


def ask_example(example_text: str, history: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    return chat_response(example_text, history)


def clear_history() -> List[Dict[str, Any]]:
    return []


# CSS Kustom: Mengatasi masalah kontras teks gelap/terang, code block hitam, box chat transparan, dan kartu pertanyaan
custom_css = """
/* Reset semua variabel tema Gradio (termasuk .dark) agar selalu terang, bersih, dan kontras */
:root, .dark, body, .gradio-container {
    --body-background-fill: #f4f5f7 !important;
    --body-text-color: #111827 !important;
    --body-text-color-subdued: #4b5563 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --border-color-primary: #e2e8f0 !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e2e8f0 !important;
    --block-label-text-color: #111827 !important;
    --input-background-fill: #ffffff !important;
    --input-text-color: #111827 !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-background-fill-hover: #f1f5f9 !important;
    --button-secondary-text-color: #111827 !important;
    --button-secondary-border-color: #e2e8f0 !important;
    --code-background-fill: #f1f5f9 !important;
    background-color: #f4f5f7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Fix tag <code> agar tidak jadi kotak hitam pekat */
code, pre {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 0.88em !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

/* Sidebar styling */
.sidebar-panel {
    background: #ffffff !important;
    border-radius: 20px !important;
    padding: 20px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
}

.sidebar-panel, .sidebar-panel * {
    color: #111827 !important;
}

.brand-pill {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #f1f5f9;
    padding: 10px 14px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1rem;
    color: #111827;
    margin-bottom: 16px;
    border: 1px solid #e2e8f0;
}

.brand-pill img {
    width: 26px;
    height: 26px;
    object-fit: contain;
}

.brand-pill span {
    color: #111827 !important;
}

/* Accordion di sidebar */
.sidebar-panel .label-wrap,
.sidebar-panel button,
.sidebar-panel details {
    background: #f8fafc !important;
    border-color: #e2e8f0 !important;
    color: #111827 !important;
}

/* Main Area Card */
.main-chat-panel {
    background: #ffffff !important;
    border-radius: 20px !important;
    padding: 22px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
}

.main-chat-panel, .main-chat-panel * {
    color: #111827 !important;
}

/* Header Top */
.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 12px;
    margin-bottom: 14px;
}

.top-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #111827 !important;
    margin: 0;
}

.upgrade-badge {
    background: #10b981;
    color: #ffffff !important;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
}

/* Welcome Center */
.welcome-hero {
    text-align: center;
    margin: 4px 0 12px 0;
}

.welcome-logo-img {
    width: 48px;
    height: 48px;
    object-fit: contain;
    margin: 0 auto 8px auto;
    display: block;
    filter: drop-shadow(0 4px 8px rgba(16, 185, 129, 0.2));
}

.welcome-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #111827 !important;
    margin: 0 0 4px 0;
}

.welcome-desc {
    color: #4b5563 !important;
    font-size: 0.88rem;
    margin: 0;
}

/* Box Chat: Transparan & selaras dengan latar */
#chatbot {
    background: transparent !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    height: 310px !important;
    margin-bottom: 14px !important;
    padding: 10px !important;
}

/* Bubble Chatbot */
#chatbot .message-wrap,
#chatbot .bubble-wrap {
    background: transparent !important;
}

#chatbot [data-testid="user"],
#chatbot .user-message {
    background-color: #f1f5f9 !important;
    color: #111827 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
}

#chatbot [data-testid="bot"],
#chatbot .bot-message {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}

#chatbot .message-content,
#chatbot .message-content * {
    color: #111827 !important;
}

/* Kotak Proses AI */
.ai-process-details {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-top: 10px !important;
    font-size: 0.86rem !important;
}

.ai-process-details summary {
    font-weight: 700 !important;
    color: #111827 !important;
    cursor: pointer !important;
    margin-bottom: 6px !important;
}

.ai-process-details,
.ai-process-details * {
    color: #1e293b !important;
}

.ai-process-details table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 8px 0 !important;
}

.ai-process-details th,
.ai-process-details td {
    border: 1px solid #cbd5e1 !important;
    padding: 6px 10px !important;
    color: #1e293b !important;
    text-align: left !important;
}

.ai-process-details th {
    background: #f1f5f9 !important;
    font-weight: 600 !important;
}

/* Kotak-kotak Contoh Pertanyaan */
.card-row {
    margin-bottom: 10px !important;
}

.prompt-card-btn {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    text-align: left !important;
    white-space: pre-line !important;
    line-height: 1.35 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.2s ease !important;
    height: auto !important;
    min-height: 62px !important;
    cursor: pointer !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 12px !important;
}

.prompt-card-btn img {
    width: 30px !important;
    height: 30px !important;
    object-fit: contain !important;
    border-radius: 6px !important;
    flex-shrink: 0 !important;
}

.prompt-card-btn,
.prompt-card-btn *,
.prompt-card-btn span {
    color: #111827 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

.prompt-card-btn:hover {
    border-color: #10b981 !important;
    background-color: #f0fdf4 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 10px rgba(16, 185, 129, 0.15) !important;
}

.prompt-card-btn:hover *,
.prompt-card-btn:hover span {
    color: #065f46 !important;
}

/* Input Bar */
.chat-input-row {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 6px 12px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    align-items: center !important;
}

.chat-input-row textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 0.95rem !important;
    color: #111827 !important;
}

.chat-input-row textarea::placeholder {
    color: #64748b !important;
}

.send-btn {
    background: #10b981 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
    transition: background 0.2s ease !important;
}

.send-btn * {
    color: #ffffff !important;
}

.send-btn:hover {
    background: #059669 !important;
}

/* Tombol Hapus Percakapan */
.btn-clear {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #111827 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    margin-top: 16px !important;
}

.btn-clear:hover {
    background: #fee2e2 !important;
    border-color: #fca5a5 !important;
    color: #991b1b !important;
}
"""

with gr.Blocks(title="ChessMate AI - Chat With AI") as demo:
    with gr.Row():
        # ==========================================
        # SIDEBAR KIRI: Panduan, Topik, Parameter
        # ==========================================
        with gr.Column(scale=1, min_width=280, elem_classes=["sidebar-panel"]):
            gr.HTML(f"""
            <div class="brand-pill">
                <img src="{LOGO_URI}" alt="Logo">
                <span>ChessMate AI</span>
            </div>
            """)

            gr.HTML(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 12px 0 8px 0;">
                <img src="{GUIDEBOOK_URI}" style="width: 22px; height: 22px; object-fit: contain;" alt="Panduan">
                <span style="font-size: 1.15rem; font-weight: 700; color: #111827;">Panduan</span>
            </div>
            """)
            gr.Markdown(
                "Aku **ChessMate AI**, asisten catur untuk pemula. "
                "Tanyakan dalam bahasa Indonesia (boleh gaul atau typo). "
                "Maksud pertanyaanmu dikenali dengan **TF-IDF + cosine similarity**, bukan API AI."
            )

            with gr.Accordion("Daftar Topik Catur", open=True):
                gr.Markdown(
                    "- Papan & posisi awal\n"
                    "- Gerakan tiap bidak\n"
                    "- Rokade, en passant, promosi\n"
                    "- Skak, skakmat, remis\n"
                    "- Nilai bidak\n"
                    "- Pembukaan catur\n"
                    "- Taktik dasar\n"
                    "- Teknik endgame\n"
                    "- Notasi catur\n"
                    "- Kesalahan pemula & cara latihan"
                )

            with gr.Accordion("Parameter Model NLP", open=False):
                gr.Markdown(
                    f"- **Ambang (tau):** {bot.confidence_threshold}\n"
                    f"- **Margin Ambigu (delta):** {bot.ambiguity_margin}\n"
                    f"- **Cakupan Kosakata Min:** {bot.min_coverage:.0%}\n"
                    f"- **Batas Typo:** {bot.typo_cutoff}"
                )

            btn_clear = gr.Button("Hapus Percakapan", elem_classes=["btn-clear"])

        # ==========================================
        # AREA UTAMA: Header, Welcome, Chat, Kotak Contoh, Input
        # ==========================================
        with gr.Column(scale=3, elem_classes=["main-chat-panel"]):
            # Top Header Bar
            gr.HTML("""
            <div class="top-header">
                <span class="top-title">Chat With AI</span>
                <span class="upgrade-badge">TF-IDF Engine</span>
            </div>
            """)

            # Welcome Hero
            gr.HTML(f"""
            <div class="welcome-hero">
                <img src="{LOGO_URI}" class="welcome-logo-img" alt="Logo">
                <h2 class="welcome-title">Welcome to ChessMate AI</h2>
                <p class="welcome-desc">Tanyakan aturan, taktik, atau strategi catur pemula. Klik contoh pertanyaan di bawah untuk memulai!</p>
            </div>
            """)

            # Area Chatbot (Transparan sesuai latar + avatar bot logo catur)
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=310,
                show_label=False,
                avatar_images=(None, str(ASSETS_DIR / "chesslogo.svg")),
            )

            # Kotak-kotak Contoh Pertanyaan (Grid 3 Kolom x 2 Baris)
            card_buttons = []
            with gr.Row(elem_classes=["card-row"]):
                for item in KARTU_CONTOH[:3]:
                    btn = gr.Button(
                        f"{item['title']}\n{item['desc']}",
                        icon=item["icon"],
                        elem_classes=["prompt-card-btn"],
                        scale=1,
                    )
                    card_buttons.append((btn, item["query"]))

            with gr.Row(elem_classes=["card-row"]):
                for item in KARTU_CONTOH[3:]:
                    btn = gr.Button(
                        f"{item['title']}\n{item['desc']}",
                        icon=item["icon"],
                        elem_classes=["prompt-card-btn"],
                        scale=1,
                    )
                    card_buttons.append((btn, item["query"]))

            # Kolom Input Pesan di Bagian Bawah
            with gr.Row(elem_classes=["chat-input-row"]):
                msg_input = gr.Textbox(
                    placeholder="Tulis pertanyaanmu tentang catur di sini... (tekan Enter untuk kirim)",
                    show_label=False,
                    scale=9,
                    lines=1,
                    max_lines=3,
                    autofocus=True,
                )
                btn_send = gr.Button("Kirim", elem_classes=["send-btn"], scale=1)

    # ==========================================
    # EVENT HANDLERS
    # ==========================================
    # Kirim dari textbox (Enter)
    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

    # Kirim dari tombol Kirim
    btn_send.click(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

    # Event saat salah satu kotak contoh pertanyaan diklik
    for btn, query in card_buttons:
        btn.click(
            fn=lambda h, q=query: ask_example(q, h),
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        )

    # Hapus percakapan
    btn_clear.click(
        fn=clear_history,
        inputs=None,
        outputs=chatbot,
    )

if __name__ == "__main__":
    demo.launch(css=custom_css, allowed_paths=[str(ASSETS_DIR)])
