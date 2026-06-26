# app.py - AI Study Assistant (Enhanced UI)

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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 1rem; }
.stat-box {
    flex: 1; background: #F8F7FF; border: 1px solid #E0DEFF;
    border-radius: 10px; padding: 12px 14px; text-align: center;
}
.stat-num { font-size: 22px; font-weight: 600; color: #534AB7; }
.stat-lbl { font-size: 11px; color: #888; margin-top: 2px; }

/* ── Answer card ── */
.answer-card {
    background: #F8F7FF; border-left: 4px solid #534AB7;
    border-radius: 0 12px 12px 0; padding: 18px 20px; margin: 12px 0;
    line-height: 1.75;
}
.answer-header {
    font-size: 12px; font-weight: 600; color: #534AB7;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}

/* ── Source chips ── */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.chip {
    font-size: 11px; background: #EEF; color: #534AB7;
    padding: 3px 10px; border-radius: 20px; border: 1px solid #DDD;
}

/* ── Quiz card ── */
.quiz-card {
    background: white; border: 1px solid #E8E8F0;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
}
.quiz-q { font-weight: 500; color: #1a1a2e; margin-bottom: 12px; font-size: 15px; }
.quiz-opt {
    padding: 8px 14px; margin: 5px 0; border-radius: 8px;
    border: 1px solid #E8E8F0; font-size: 13px; color: #444;
    background: #FAFAFA;
}
.quiz-opt.correct {
    background: #F0FFF4; border-color: #68D391; color: #276749; font-weight: 500;
}
.quiz-num {
    display: inline-block; background: #534AB7; color: white;
    font-size: 11px; font-weight: 600; padding: 2px 8px;
    border-radius: 20px; margin-bottom: 10px;
}

/* ── Section header ── */
.section-header {
    font-size: 20px; font-weight: 600; color: #1a1a2e;
    margin-bottom: 6px; display: flex; align-items: center; gap: 8px;
}
.section-sub { font-size: 13px; color: #888; margin-bottom: 20px; }

/* ── Summary card ── */
.summary-card {
    background: white; border: 1px solid #E8E8F0;
    border-radius: 12px; padding: 20px 24px; line-height: 1.8;
    font-size: 14px; color: #333;
}

/* ── Chat history item ── */
.history-q {
    font-size: 12px; font-weight: 600; color: #534AB7;
    margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em;
}

/* ── Badge ── */
.rag-badge {
    display: inline-block; background: #EEEDFE; color: #534AB7;
    font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; margin-left: 8px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 60px 20px; color: #aaa;
}
.empty-icon { font-size: 48px; margin-bottom: 12px; }
.empty-title { font-size: 18px; font-weight: 500; color: #555; margin-bottom: 8px; }
.empty-sub { font-size: 13px; line-height: 1.6; }

/* ── Progress steps ── */
.step-done { color: #38A169; font-weight: 500; font-size: 13px; }
.step-active { color: #534AB7; font-weight: 500; font-size: 13px; }
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
    "active_tab": "qa",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Study Assistant")
    st.markdown('<span class="rag-badge">Gemini + RAG</span>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your PDF notes", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        st.markdown("**Processing steps:**")
        s1 = st.empty(); s2 = st.empty(); s3 = st.empty()

        s1.markdown('<p class="step-active">⏳ Extracting text...</p>', unsafe_allow_html=True)
        text, ocr_used = extract_text_from_pdf(uploaded_file)

        if not text:
            st.error("❌ Could not extract text from this PDF.")
        else:
            pages = text.count("--- Page")
            if ocr_used:
                s1.markdown('<p class="step-done">✅ Text extracted (via OCR 🔍)</p>', unsafe_allow_html=True)
            else:
                s1.markdown('<p class="step-done">✅ Text extracted</p>', unsafe_allow_html=True)

            s2.markdown('<p class="step-active">⏳ Chunking text...</p>', unsafe_allow_html=True)
            chunks = get_text_chunks(text)
            s2.markdown(f'<p class="step-done">✅ {len(chunks)} chunks created</p>', unsafe_allow_html=True)

            s3.markdown('<p class="step-active">⏳ Building vector DB...</p>', unsafe_allow_html=True)
            result = create_vector_store(chunks)

            if result:
                s3.markdown('<p class="step-done">✅ Knowledge base ready</p>', unsafe_allow_html=True)
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.num_chunks = len(chunks)
                st.session_state.num_pages = pages
                st.session_state.chat_history = []
                st.session_state.summary = ""
                st.session_state.quiz_raw = ""
                st.success("✅ Ready! Use the tabs to explore.")
            else:
                s3.markdown("❌ Vector DB failed", unsafe_allow_html=True)

    # Stats panel
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-num">{st.session_state.num_pages}</div>
                <div class="stat-lbl">Pages</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">{st.session_state.num_chunks}</div>
                <div class="stat-lbl">Chunks</div>
            </div>
        </div>
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-num">768</div>
                <div class="stat-lbl">Vector dims</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">5</div>
                <div class="stat-lbl">Top-K</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"📄 **{st.session_state.pdf_name}**")
        st.markdown("---")

    st.markdown("### How it works")
    st.markdown("""
1. Upload your PDF
2. AI reads & indexes it
3. Ask questions in natural language
4. Get answers sourced from your notes
    """)

# ── Main area ──────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📚</div>
        <div class="empty-title">Upload a PDF to get started</div>
        <div class="empty-sub">
            Upload your lecture notes, textbook chapters, or study guides<br>
            from the sidebar and your AI study assistant will be ready in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("💬 **Ask Questions**\nGet instant answers grounded in your notes")
    with col2:
        st.info("📄 **Summaries**\nKey topics and takeaways in seconds")
    with col3:
        st.info("📝 **Quiz Yourself**\nAuto-generated MCQs to test your knowledge")

else:
    # Tab navigation
    tab_qa, tab_summary, tab_quiz = st.tabs(["💬 Ask Questions", "📄 Summary", "📝 Quiz"])

    # ── Q&A Tab ────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown('<div class="section-header">💬 Ask Questions</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Questions are answered using only your uploaded document</div>', unsafe_allow_html=True)

        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_q = st.text_input(
                "Your question",
                placeholder="e.g. What is the main objective of this project?",
                label_visibility="collapsed"
            )
        with col_btn:
            ask = st.button("Ask →", type="primary", use_container_width=True)

        if ask:
            if not user_q.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching your notes..."):
                    ans = answer_question(user_q)
                st.session_state.chat_history.insert(0, (user_q, ans))

        if st.session_state.chat_history:
            if len(st.session_state.chat_history) > 1:
                if st.button("Clear history", type="secondary"):
                    st.session_state.chat_history = []
                    st.rerun()

            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-header">📌 {q}</div>
                    {a}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding: 40px; color: #aaa;">
                <div style="font-size:32px; margin-bottom:8px;">💬</div>
                <div style="font-size:14px;">Ask your first question above</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Summary Tab ────────────────────────────────────────────────────────
    with tab_summary:
        st.markdown('<div class="section-header">📄 Document Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Structured overview of key topics and concepts</div>', unsafe_allow_html=True)

        if st.button("✨ Generate Summary", type="primary"):
            with st.spinner("Analyzing your document..."):
                st.session_state.summary = generate_summary()

        if st.session_state.summary:
            st.markdown(f'<div class="summary-card">{st.session_state.summary}</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name=f"summary_{st.session_state.pdf_name}.txt",
                mime="text/plain"
            )
        else:
            st.markdown("""
            <div style="text-align:center; padding: 40px; color: #aaa;">
                <div style="font-size:32px; margin-bottom:8px;">📄</div>
                <div style="font-size:14px;">Click Generate Summary to create an overview</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Quiz Tab ───────────────────────────────────────────────────────────
    with tab_quiz:
        st.markdown('<div class="section-header">📝 Practice Quiz</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">AI-generated multiple choice questions from your notes</div>', unsafe_allow_html=True)

        col_slider, col_genbtn = st.columns([3, 1])
        with col_slider:
            num_q = st.slider("Number of questions", 3, 10, 5, label_visibility="visible")
        with col_genbtn:
            st.markdown("<br>", unsafe_allow_html=True)
            gen_quiz = st.button("🎯 Generate Quiz", type="primary", use_container_width=True)

        if gen_quiz:
            with st.spinner("Creating your quiz..."):
                st.session_state.quiz_raw = generate_quiz(num_questions=num_q)

        if st.session_state.quiz_raw:
            # Parse and render quiz beautifully
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

                q_text = lines[0].split('.', 1)[-1].strip() if lines else ""
                opts = [l.strip() for l in lines[1:] if l.strip().startswith(('A)', 'B)', 'C)', 'D)'))]
                correct_line = next((l for l in lines if '✅' in l or 'Correct Answer' in l), "")

                correct_letter = ""
                if correct_line:
                    for ch in ['A', 'B', 'C', 'D']:
                        if f': {ch}' in correct_line or f'Answer: {ch}' in correct_line:
                            correct_letter = ch
                            break

                with st.container():
                    st.markdown(f'<div class="quiz-num">Q{idx+1}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="quiz-q">{q_text}</div>', unsafe_allow_html=True)

                    for opt in opts:
                        letter = opt[0] if opt else ""
                        css_class = "quiz-opt correct" if letter == correct_letter else "quiz-opt"
                        prefix = "✅ " if letter == correct_letter else ""
                        st.markdown(f'<div class="{css_class}">{prefix}{opt}</div>', unsafe_allow_html=True)

                    if correct_line:
                        explanation = correct_line.split('-', 1)[-1].strip() if '-' in correct_line else ""
                        if explanation:
                            st.caption(f"💡 {explanation}")

                    st.markdown("<br>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Quiz",
                data=st.session_state.quiz_raw,
                file_name=f"quiz_{st.session_state.pdf_name}.txt",
                mime="text/plain"
            )
        else:
            st.markdown("""
            <div style="text-align:center; padding: 40px; color: #aaa;">
                <div style="font-size:32px; margin-bottom:8px;">📝</div>
                <div style="font-size:14px;">Click Generate Quiz to create practice questions</div>
            </div>
            """, unsafe_allow_html=True)
