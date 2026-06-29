# test_cloud.py - Run this to test all imports work correctly
print("Testing imports...")

try:
    from config import get_embeddings, get_llm, GROQ_API_KEY
    print("✅ config.py OK")
    print(f"   API key found: {bool(GROQ_API_KEY)}")
except Exception as e:
    print(f"❌ config.py FAILED: {e}")

try:
    from pdf_handler import extract_text_from_pdf
    print("✅ pdf_handler.py OK")
except Exception as e:
    print(f"❌ pdf_handler.py FAILED: {e}")

try:
    from text_chunker import get_text_chunks
    print("✅ text_chunker.py OK")
except Exception as e:
    print(f"❌ text_chunker.py FAILED: {e}")

try:
    from vector_store import create_vector_store, get_relevant_chunks
    print("✅ vector_store.py OK")
except Exception as e:
    print(f"❌ vector_store.py FAILED: {e}")

try:
    from chatbot import answer_question
    print("✅ chatbot.py OK")
except Exception as e:
    print(f"❌ chatbot.py FAILED: {e}")

try:
    from summarizer import generate_summary
    print("✅ summarizer.py OK")
except Exception as e:
    print(f"❌ summarizer.py FAILED: {e}")

try:
    from quiz_generator import generate_quiz
    print("✅ quiz_generator.py OK")
except Exception as e:
    print(f"❌ quiz_generator.py FAILED: {e}")

try:
    emb = get_embeddings()
    vec = emb.embed_query("test")
    print(f"✅ Embeddings working! Vector size: {len(vec)}")
except Exception as e:
    print(f"❌ Embeddings FAILED: {e}")

print("\n🎉 All tests complete!")
