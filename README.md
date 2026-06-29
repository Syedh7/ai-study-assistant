# 📚 AI Study Assistant for Engineering Students

An AI-powered study tool that lets students upload PDF notes and interact with them using natural language — powered by Google Gemini, LangChain, and FAISS.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 PDF Upload | Upload any text-based PDF study material |
| 💬 Question Answering | Ask questions, get answers grounded in your notes |
| 📝 Summary Generation | Get a structured overview of key topics |
| 🎯 Quiz Generator | Auto-generate MCQ quizzes from your material |
| 🧠 RAG Architecture | Retrieval-Augmented Generation for accurate answers |

---

## 🏗️ Architecture

```
User uploads PDF
      ↓
Extract Text (pypdf)
      ↓
Chunk Text (LangChain RecursiveCharacterTextSplitter)
      ↓
Generate Embeddings (Gemini Embedding-001)
      ↓
Store in FAISS Vector DB
      ↓
User asks a question
      ↓
Retrieve top-5 relevant chunks (semantic search)
      ↓
Send context + question to Gemini 1.5 Flash
      ↓
Display answer in Streamlit UI
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini 1.5 Flash
- **Embeddings:** Gemini Embedding-001
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Framework:** LangChain
- **PDF Parsing:** pypdf

---

## 📁 Project Structure

```
ai_study_assistant/
├── app.py              # Main Streamlit UI
├── config.py           # Central config (API key, model names)
├── pdf_handler.py      # PDF text extraction
├── text_chunker.py     # Split text into chunks
├── vector_store.py     # FAISS create/load/search
├── chatbot.py          # RAG question answering
├── summarizer.py       # Document summarization
├── quiz_generator.py   # MCQ quiz generation
├── requirements.txt    # Dependencies
├── .env                # API key (NOT committed to Git)
└── .gitignore
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-study-assistant.git
cd ai-study-assistant

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
# Create a .env file with:
# GOOGLE_API_KEY=your_key_here

# 5. Run the app
streamlit run app.py
```

---

## 🔑 API Key Setup

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a free API key
3. Add to `.env`: `GOOGLE_API_KEY=your_key`

---

## 👨‍💻 Built For

B.Tech AIML — AI Study Assistant using RAG
