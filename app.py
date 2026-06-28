# app.py - AI Study Assistant (Mobile-Friendly UI)

import streamlit as st
from pdf_handler import extract_text_from_pdf
from text_chunker import get_text_chunks
from vector_store import create_vector_store
from chatbot import answer_question
from summarizer import generate_summary
from quiz_generator import generate_quiz

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered",  # centered works better on mobile
    initial_sidebar_state="collapsed"  # hide sidebar by default
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container { padding: 1rem 1rem 2rem 1rem !important; max-width: 800px; }

/* ── Header ── */
.app-header {
    text-align: center; padding: 24px 16px 16px 16px;
    background: linear-gradient(135deg, #1a1a2e 0%, #534AB7 100%);
    border-radius: 16px; margin-bottom: 20px; color: white;
}
.app-title { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
.app-sub { font-size: 13px; opacity: 0.8; }
.app-badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    padding: 3px 12px; border-radius: 20px; font-size: 11px;
    margin-top: 8px; font-weight: 600;
}

/* ── Upload card ── */
.upload-card {
    background: #F8F7FF; border: 2px dashed #C4C0F0;
    border-radius: 16px; padding: 24px 20px; text-align: center;
    margin-bottom: 16px;
}
.upload-title { font-size: 16px; font-weight: 600; color: #1a1a2e; margin-bottom: 6px; }
.upload-sub { font-size: 12px; color: #888; margin-bottom: 16px; }

/* ── Status card ── */
.status-card {
    background: #F0FFF4; border: 1px solid #68D391;
    border-radius: 12px; padding: 14px 16px; margin-bottom: 16px;
}
.status-title { font-size: 13px; font-weight: 600; color: #276749; }

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0; }
.stat-box {
    background: white; border: 1px solid #E0DEFF;
    border-radius: 10px; padding: 10px 6px; text-align: center;
}
.stat-num { font-size: 18px; font-weight: 700; color: #534AB7; }
.stat-lbl { font-size: 10px; color: #888; margin-top: 2px; }

/* ── Feature cards ── */
.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 16px 0; }
.feat-card {
    background: white; border: 1px solid #E8E8F0;
    border-radius: 12px; padding: 14px 10px; text-align: center;
}
.feat-icon { font-size: 24px; margin-bottom: 6px; }
.feat-title { font-size: 12px; font-weight: 600; color: #1a1a2e; }
.feat-desc { font-size: 11px; color: #888; margin-top: 3px; }

/* ── Answer card ── */
.answer-card {
    background: #F8F7FF; border-left: 4px solid #534AB7;
    border-radius: 0 12px 12px 0; padding: 16px;
    margin: 10px 0; line-height: 1.75; font-size: 14px;
}
.answer-q {
    font-size: 11px; font-weight: 700; color: #534AB7;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 8px;
}

/* ── Quiz card ── */
.quiz-num {
    display: inline-block; background: #534AB7; color: white;
    font-size: 11px; font-weight: 700; padding: 2px 10px;
    border-radius: 20px; margin-bottom: 8px;
}
.quiz-q { font-weight: 600; color: #1a1a2e; margin-bottom: 10px; font-size: 14px; }
.quiz-opt {
    padding: 8px 12px; margin: 4px 0; border-radius: 8px;
    border: 1px solid #E8E8F0; font-size: 13px; color: #444;
    background: #FAFAFA;
}
.quiz-opt.correct {
    background: #F0FFF4; border-color: #68D391;
    color: #276749; font-weight: 600;
}

/* ── Summary card ── */
.summary-card {
    background: white; border: 1px solid #E8E8F0;
    border-radius: 12px; padding: 18px; line-height: 1.8;
    font-size: 14px; color: #333;
}

/* ── Step indicator ── */
.step-done { color: #38A169; font-weight: 600; font-size: 13px; margin: 4px 0; }
.step-active { color: #534AB7; font-weight: 600; font-size: 13px; margin: 4px 0; }

/* ── Section header ── */
.sec-header { font-size: 18px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }
.sec-sub { font-size: 13px; color: #888; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in {
    "pdf_processed": False,
    "pdf_name": "",
    "num_chunks": 0,
    "num_pages": 0,
    "chat_history": [],
    "summary": "",
    "quiz_raw": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">📚 AI Study Assistant</div>
    <div class="app-sub">Study smarter with AI — powered by RAG</div>
    <div class="app-badge">Groq Llama 3.3 + FAISS + LangChain</div>
</div>
""", unsafe_allow_html=True)

# ── PDF Upload — always visible on main screen ─────────────────────────────
if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-title">📄 Upload Your Study Material</div>
        <div class="upload-sub">Supports any PDF — lecture notes, textbooks, lab reports</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown("---")
        s1 = st.empty(); s2 = st.empty(); s3 = st.empty()

        s1.markdown('<p class="step-active">⏳ Extracting text...</p>', unsafe_allow_html=True)
        text, ocr_used = extract_text_from_pdf(uploaded_file)

        if not text:
            st.error("❌ Could not extract text. Try a different PDF.")
        else:
            pages = text.count("--- Page")
            label = "✅ Text extracted (OCR 🔍)" if ocr_used else "✅ Text extracted"
            s1.markdown(f'<p class="step-done">{label}</p>', unsafe_allow_html=True)

            s2.markdown('<p class="step-active">⏳ Building chunks...</p>', unsafe_allow_html=True)
            chunks = get_text_chunks(text)
            s2.markdown(f'<p class="step-done">✅ {len(chunks)} chunks created</p>', unsafe_allow_html=True)

            s3.markdown('<p class="step-active">⏳ Building AI knowledge base...</p>', unsafe_allow_html=True)
            result = create_vector_store(chunks)

            if result:
                s3.markdown('<p class="step-done">✅ Knowledge base ready!</p>', unsafe_allow_html=True)
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.num_chunks = len(chunks)
                st.session_state.num_pages = pages
                st.session_state.chat_history = []
                st.session_state.summary = ""
                st.session_state.quiz_raw = ""
                st.rerun()
            else:
                s3.markdown('<p style="color:red;">❌ Knowledge base failed</p>', unsafe_allow_html=True)

    # Feature cards shown before upload
    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">💬</div>
            <div class="feat-title">Ask Questions</div>
            <div class="feat-desc">Instant answers from your notes</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📄</div>
            <div class="feat-title">Summaries</div>
            <div class="feat-desc">Key topics in seconds</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📝</div>
            <div class="feat-title">Quiz</div>
            <div class="feat-desc">Auto-generated MCQs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── PDF loaded — show stats + tabs ────────────────────────────────────

    # Status bar
    st.markdown(f"""
    <div class="status-card">
        <div class="status-title">✅ {st.session_state.pdf_name}</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-num">{st.session_state.num_pages}</div>
            <div class="stat-lbl">Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">{st.session_state.num_chunks}</div>
            <div class="stat-lbl">Chunks</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">768</div>
            <div class="stat-lbl">Dims</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">Top-5</div>
            <div class="stat-lbl">RAG-K</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New PDF button
    if st.button("📤 Upload New PDF", use_container_width=True):
        for key in ["pdf_processed", "pdf_name", "num_chunks",
                    "num_pages", "chat_history", "summary", "quiz_raw"]:
            st.session_state[key] = False if key == "pdf_processed" else "" if isinstance(st.session_state[key], str) else 0 if isinstance(st.session_state[key], int) else []
        st.rerun()

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_qa, tab_summary, tab_quiz = st.tabs(["💬 Ask", "📄 Summary", "📝 Quiz"])

    # ── Q&A Tab ───────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown('<div class="sec-header">💬 Ask Questions</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Answers grounded in your document only</div>', unsafe_allow_html=True)

        user_q = st.text_input(
            "Question",
            placeholder="e.g. What is the main objective?",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            ask = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if ask:
            if not user_q.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching your notes..."):
                    ans = answer_question(user_q)
                st.session_state.chat_history.insert(0, (user_q, ans))

        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-q">📌 {q}</div>
                {a}
            </div>
            """, unsafe_allow_html=True)

    # ── Summary Tab ───────────────────────────────────────────────────────
    with tab_summary:
        st.markdown('<div class="sec-header">📄 Document Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Structured overview of key topics</div>', unsafe_allow_html=True)

        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            with st.spinner("Analyzing document..."):
                st.session_state.summary = generate_summary()

        if st.session_state.summary:
            st.markdown(f'<div class="summary-card">{st.session_state.summary}</div>',
                       unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name=f"summary_{st.session_state.pdf_name}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ── Quiz Tab ──────────────────────────────────────────────────────────
    with tab_quiz:
        st.markdown('<div class="sec-header">📝 Practice Quiz</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Auto-generated MCQs from your notes</div>', unsafe_allow_html=True)

        num_q = st.slider("Number of questions", 3, 10, 5)

        if st.button("🎯 Generate Quiz", type="primary", use_container_width=True):
            with st.spinner("Creating quiz..."):
                st.session_state.quiz_raw = generate_quiz(num_questions=num_q)

        if st.session_state.quiz_raw:
            raw = st.session_state.quiz_raw
            questions = []
            current = []
            for line in raw.split('\n'):
                if line.strip().startswith('Q') and '.' in line[:4] and current:
                    questions.append('\n'.join(current))
                    current = [line]
                elif '--- END' in line:
                    if current:
                        questions.append('\n'.join(current))
                    current = []
                else:
                    current.append(line)
            if current:
                questions.append('\n'.join(current))

            for idx, q_block in enumerate(questions):
                if not q_block.strip():
                    continue
                lines = [l for l in q_block.strip().split('\n') if l.strip()]
                if not lines:
                    continue

                q_text = lines[0].split('.', 1)[-1].strip()
                opts = [l.strip() for l in lines[1:] if l.strip().startswith(('A)', 'B)', 'C)', 'D)'))]
                correct_line = next((l for l in lines if '✅' in l or 'Correct Answer' in l), "")

                correct_letter = ""
                if correct_line:
                    for ch in ['A', 'B', 'C', 'D']:
                        if f': {ch}' in correct_line or f'Answer: {ch}' in correct_line:
                            correct_letter = ch
                            break

                st.markdown(f'<div class="quiz-num">Q{idx+1}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="quiz-q">{q_text}</div>', unsafe_allow_html=True)

                for opt in opts:
                    letter = opt[0] if opt else ""
                    css_class = "quiz-opt correct" if letter == correct_letter else "quiz-opt"
                    prefix = "✅ " if letter == correct_letter else ""
                    st.markdown(f'<div class="{css_class}">{prefix}{opt}</div>', unsafe_allow_html=True)

                if correct_line and '-' in correct_line:
                    exp = correct_line.split('-', 1)[-1].strip()
                    if exp:
                        st.caption(f"💡 {exp}")

                st.markdown("<br>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Quiz",
                data=st.session_state.quiz_raw,
                file_name=f"quiz_{st.session_state.pdf_name}.txt",
                mime="text/plain",
                use_container_width=True
            )
