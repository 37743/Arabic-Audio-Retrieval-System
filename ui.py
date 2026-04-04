import streamlit as st
import requests
import base64
import os

base_dir = os.path.dirname(__file__)

st.set_page_config(
    page_title="Arabic Audio Semantic Search",
    page_icon="🎙️",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/semantic_query"

logo_path = os.path.join(base_dir, "ui", "textures", "ej.png")
bg_path = os.path.join(base_dir, "ui", "textures", "background.png")

# Background size control
BACKGROUND_SIZE = "100% auto"

# Use the same icon for both user and assistant
CHAT_AVATAR = logo_path if os.path.exists(logo_path) else "🎙️"


def get_base64_image(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()


logo_image = get_base64_image(logo_path)
bg_image = get_base64_image(bg_path)

st.markdown(
    f"""
<style>
:root {{
    --primary-red: #ea232b;
    --text-black: #000000;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 12px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(234, 35, 43, 0.35); border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(234, 35, 43, 0.65); }}

/* Hide Streamlit chrome */
[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
header,
footer {{
    display: none !important;
}}

/* App background */
.stApp {{
    {"background-image: url('data:image/png;base64," + bg_image + "');" if bg_image else "background-image: linear-gradient(180deg, #fdfdfd 0%, #f6f7fb 100%);"}
    background-position: center top;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: {BACKGROUND_SIZE};
    color: var(--text-black) !important;
}}

/* White fade overlay from top to bottom */
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(
        180deg,
        rgba(255, 255, 255, 0.96) 0%,
        rgba(255, 255, 255, 0.78) 28%,
        rgba(255, 255, 255, 0.48) 62%,
        rgba(255, 255, 255, 0.12) 100%
    );
    z-index: 0;
    pointer-events: none;
}}

/* Layout */
.block-container {{
    position: relative;
    z-index: 2;
    padding-top: 2rem !important;
    padding-bottom: 120px !important;
    max-width: 900px !important;
}}

/* Header */
.header-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1.6rem;
    background: rgba(255, 255, 255, 0.82);
    border-radius: 22px;
    backdrop-filter: blur(12px);
    border: 2px solid var(--primary-red) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    transition: transform 0.25s ease;
}}

.header-container:hover {{
    transform: translateY(-4px);
}}

.title {{
    font-size: 34px;
    font-weight: 800;
    color: var(--text-black) !important;
    margin-top: 14px;
    margin-bottom: 6px;
    text-align: center;
    letter-spacing: 0.8px;
}}

.subtitle {{
    font-size: 15px;
    color: var(--text-black) !important;
    text-align: center;
    margin: 0;
    opacity: 0.82;
}}

.logo {{
    display: block;
    width: 110px;
    animation: bounce 1.5s infinite cubic-bezier(0.28, 0.84, 0.42, 1);
    filter: drop-shadow(0 15px 10px rgba(0,0,0,0.12));
}}

@keyframes bounce {{
    0%, 100% {{ transform: translateY(0) scale(1); }}
    50% {{ transform: translateY(-16px) scale(1.03); }}
}}

@keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* Global text */
html, body, p, span, div, label, small, li {{
    color: var(--text-black) !important;
}}

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {{
    direction: rtl !important;
    text-align: right !important;
    color: var(--text-black) !important;
}}

/* Chat messages */
[data-testid="stChatMessage"] {{
    direction: rtl !important;
    background: rgba(255, 255, 255, 0.94) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-radius: 20px !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
    animation: slideUp 0.35s ease forwards;
    color: var(--text-black) !important;
}}

[data-testid="stChatMessage"] * {{
    color: var(--text-black) !important;
}}

[data-testid="stChatMessage"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.07) !important;
}}

[data-testid="stChatMessage"][aria-label="Chat message from user"] {{
    border: 2px solid var(--primary-red) !important;
    box-shadow: 0 4px 15px rgba(234, 35, 43, 0.18) !important;
}}

/* Bottom/input area */
[data-testid="stBottom"],
[data-testid="stBottomBlock"],
.stAppBottomBlock {{
    background: transparent !important;
    background-color: transparent !important;
}}

[data-testid="stBottom"] > div,
[data-testid="stBottomBlock"] > div {{
    background: transparent !important;
    background-color: transparent !important;
}}

[data-testid="stChatInput"] {{
    background-color: transparent !important;
    padding: 5px 15px !important;
}}

/* Outer chat input shell */
[data-testid="stChatInput"] > div {{
    background-color: #ffffff !important;
    border: 2px solid var(--primary-red) !important;
    border-radius: 30px !important;
    box-shadow: 0 -5px 25px rgba(0, 0, 0, 0.08) !important;
    padding: 5px 15px !important;
}}

[data-testid="stChatInput"] > div:focus-within {{
    box-shadow: 0 0 25px rgba(234, 35, 43, 0.22) !important;
    border-color: var(--primary-red) !important;
}}

/* BaseWeb textarea wrappers */
[data-baseweb="textarea"] {{
    background-color: #ffffff !important;
    border-radius: 24px !important;
}}

[data-baseweb="textarea"] > div {{
    background-color: #ffffff !important;
    border-radius: 24px !important;
}}

[data-baseweb="textarea"] textarea {{
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: var(--primary-red) !important;
}}

/* Streamlit textarea selectors */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:active {{
    direction: rtl !important;
    text-align: right !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: var(--primary-red) !important;
    box-shadow: none !important;
    border: none !important;
}}

textarea[aria-label="ابحث داخل الملفات الصوتية..."] {{
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: var(--primary-red) !important;
    padding: 12px 0px 12px 10px !important;
}}

textarea[aria-label="ابحث داخل الملفات الصوتية..."]::placeholder {{
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 0.55 !important;
}}

[data-testid="stChatInput"] button {{
    display: none !important;
}}

/* Hide helper banner / instructions under input */
[data-testid="InputInstructions"],
[data-testid="InputInstructions"] *,
[data-testid="stChatInputInstructions"],
[data-testid="stChatInputInstructions"] *,
.stChatInputInstructions,
.stChatInputInstructions *,
[data-testid="stChatInput"] small,
[data-testid="stChatInput"] p,
[data-testid="stChatInput"] + div,
[data-testid="stChatInput"] + div *,
section[data-testid="stChatInput"] + div,
section[data-testid="stChatInput"] + div * {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}

/* Result cards */
.result-card {{
    background: rgba(255, 255, 255, 0.90);
    border-right: 4px solid var(--primary-red);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    color: var(--text-black) !important;
}}

.result-title {{
    font-weight: 800;
    color: var(--primary-red) !important;
    margin-bottom: 6px;
    font-size: 17px;
}}

.result-meta {{
    color: var(--text-black) !important;
    font-size: 14px;
    margin-bottom: 10px;
}}

.result-chunk {{
    color: var(--text-black) !important;
    line-height: 1.9;
}}

.section-label {{
    color: var(--primary-red) !important;
    font-weight: 700;
}}

/* Expander */
[data-testid="stExpander"] {{
    background-color: rgba(255, 255, 255, 0.76) !important;
    border: 1px solid rgba(234, 35, 43, 0.25) !important;
    border-radius: 12px !important;
    color: var(--text-black) !important;
}}

[data-testid="stExpander"] * {{
    color: var(--text-black) !important;
}}

[data-testid="stExpander"] details,
[data-testid="stExpander"] details[open],
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details summary:hover {{
    background-color: transparent !important;
}}

[data-testid="stExpander"] details summary p {{
    color: var(--primary-red) !important;
    font-weight: bold !important;
    direction: rtl !important;
    text-align: right !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header-container">
    {f'<img class="logo" src="data:image/png;base64,{logo_image}">' if logo_image else ''}
    <div class="title">البحث الدلالي في الملفات الصوتية</div>
    <p class="subtitle">ابحث بالمعنى داخل التفريغات النصية واسترجع أكثر الملفات ارتباطًا</p>
</div>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=CHAT_AVATAR):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("results"):
            for item in message["results"]:
                file_name = item.get("file_name", "Unknown")
                score = item.get("score", 0.0)
                best_chunk = item.get("best_chunk", "")
                full_transcription = item.get("full_transcription", "")
                chunk_index = item.get("chunk_index", "-")

                st.markdown(
                    f"""
<div class="result-card">
    <div class="result-title">🎧 {file_name}</div>
    <div class="result-meta">
        <span class="section-label">درجة التشابه:</span> {score:.4f}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="section-label">المقطع:</span> {chunk_index}
    </div>
    <div class="result-chunk">
        <span class="section-label">أفضل مقطع مطابق:</span><br>
        {best_chunk}
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                with st.expander(f"عرض التفريغ الكامل للملف: {file_name}"):
                    st.markdown(full_transcription if full_transcription else "لا يوجد تفريغ كامل.")

prompt = st.chat_input("ابحث داخل الملفات الصوتية...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=CHAT_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=CHAT_AVATAR):
        with st.spinner("جارٍ تنفيذ البحث الدلالي..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": prompt, "top_k": 5},
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])

                    summary = f"تم العثور على {len(results)} نتيجة مرتبطة بسؤالك."
                    st.markdown(summary)

                    for item in results:
                        file_name = item.get("file_name", "Unknown")
                        score = item.get("score", 0.0)
                        best_chunk = item.get("best_chunk", "")
                        full_transcription = item.get("full_transcription", "")
                        chunk_index = item.get("chunk_index", "-")

                        st.markdown(
                            f"""
<div class="result-card">
    <div class="result-title">🎧 {file_name}</div>
    <div class="result-meta">
        <span class="section-label">درجة التشابه:</span> {score:.4f}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="section-label">المقطع:</span> {chunk_index}
    </div>
    <div class="result-chunk">
        <span class="section-label">أفضل مقطع مطابق:</span><br>
        {best_chunk}
    </div>
</div>
""",
                            unsafe_allow_html=True,
                        )

                        with st.expander(f"عرض التفريغ الكامل للملف: {file_name}"):
                            st.markdown(full_transcription if full_transcription else "لا يوجد تفريغ كامل.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": summary,
                        "results": results
                    })

                else:
                    st.error("⚠️ حدث خطأ في الاتصال بالخادم.")

            except Exception as e:
                st.error(f"⚠️ خطأ في الاتصال: {e}")