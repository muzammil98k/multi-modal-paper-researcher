import os
import shutil
import time
import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import pandas as pd

# --- Page Config ---
# Must be the first Streamlit command
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# --- Load API Key ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CORE FUNCTIONS ---


def process_pdf_and_extract_content(uploaded_file):
    """
    Extracts text, images, and tables from an uploaded PDF file.
    """
    if os.path.exists("images"):
        shutil.rmtree("images")
    os.makedirs("images")

    try:
        file_bytes = uploaded_file.getvalue()
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

        full_text, image_paths, extracted_tables = "", [], []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    if not isinstance(img, (list, tuple)) or len(img) == 0:
                        continue
                    xref = img[0]
                    if xref <= 0:
                        continue
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_path = f"images/page{page_num+1}_img{img_index+1}.png"
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)
                    image_paths.append(image_path)
                except Exception as e:
                    st.warning(
                        f"Could not extract image {img_index+1} on page {page_num+1}. Skipping. Reason: {e}")
                    continue

            table_list = page.find_tables()
            for table in table_list:
                df = table.to_pandas()
                if not df.empty:
                    extracted_tables.append(df.to_markdown())

        return full_text, image_paths, extracted_tables
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return None, None, None


def chunk_text(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)


def embed_and_store_text(chunks, client):
    """Embeds text chunks and stores them in ChromaDB."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = client.get_or_create_collection("pdf_text_embeddings")
    collection.add(embeddings=model.encode(chunks), documents=chunks, ids=[
                   f"chunk_{i}" for i in range(len(chunks))])
    return collection


def embed_and_store_images(image_paths, client):
    """Embeds images and stores them in ChromaDB."""
    model = SentenceTransformer('clip-ViT-B-32')
    collection = client.get_or_create_collection("pdf_image_embeddings")
    for img_path in image_paths:
        img_embedding = model.encode(Image.open(img_path))
        collection.add(embeddings=[img_embedding.tolist()], documents=[
                       img_path], ids=[img_path])
    return collection


def embed_and_store_tables(tables, client):
    """Summarizes, embeds, and stores tables in ChromaDB."""
    text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarization_model = genai.GenerativeModel('models/gemini-pro-latest')
    collection = client.get_or_create_collection("pdf_table_embeddings")
    for i, table_md in enumerate(tables):
        prompt = f"Provide a concise, one-sentence summary of the following table's main topic or the type of data it presents. Table:\n{table_md}"
        try:
            summary = summarization_model.generate_content(prompt).text
            summary_embedding = text_embed_model.encode([summary])
            collection.add(embeddings=[summary_embedding[0].tolist()], documents=[
                           summary], metadatas=[{"original_table": table_md}], ids=[f"table_{i}"])
            time.sleep(1)
        except Exception as e:
            st.warning(f"Could not summarize or embed table {i+1}: {e}")
            if "429" in str(e):
                st.warning("Rate limit hit. Waiting for 20 seconds...")
                time.sleep(20)
            continue
    return collection


def get_final_answer(user_question, text_collection, image_collection, table_collection):
    """Performs hybrid retrieval and generates an answer using a multi-modal LLM with a strict prompt."""
    text_embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    image_embed_model = SentenceTransformer('clip-ViT-B-32')

    text_question_embedding = text_embed_model.encode([user_question])
    image_question_embedding = image_embed_model.encode([user_question])

    text_results = text_collection.query(
        query_embeddings=text_question_embedding, n_results=5) if text_collection else {'documents': [[]]}
    image_results = image_collection.query(
        query_embeddings=image_question_embedding, n_results=2) if image_collection else {'documents': [[]]}
    table_results = table_collection.query(query_embeddings=text_question_embedding, n_results=2) if table_collection else {
        'metadatas': [[]], 'documents': [[]]}

    context_text = "\n---\n".join(text_results['documents'][0])
    retrieved_image_paths = image_results['documents'][0]
    retrieved_images = [Image.open(img_path)
                        for img_path in retrieved_image_paths]
    retrieved_tables_md = [meta['original_table']
                           for meta in table_results.get('metadatas', [[]])[0]]
    context_tables = "\n\n".join(retrieved_tables_md)

    prompt_parts = [
        "**TASK:** You are a component in a document analysis system. Your sole function is to answer a user's question based ONLY on the provided context from a scientific paper.\n\n",
        "**STRICT RULES:**\n",
        "1. You MUST derive your answer exclusively from the 'CONTEXT' provided below.\n",
        "2. Do NOT use any external knowledge, personal opinions, or general information.\n",
        "3. If the answer is not found in the context, you MUST reply with the exact phrase: 'The answer could not be found in the provided document.'\n",
        "4. Do not explain your reasoning or mention that you are an AI. Just provide the answer to the user's question.\n\n",
        "**CONTEXT:**\n",
        "--- TEXT SNIPPETS ---\n", context_text,
        "\n--- TABLES ---\n", context_tables,
        "\n--- IMAGES ---\n", "[Images are provided as direct input]\n\n",
        "**USER QUESTION:**\n", user_question,
        "\n\n**ANSWER:**"
    ]
    prompt_parts.extend(retrieved_images)

    model_gen = genai.GenerativeModel('models/gemini-pro-latest')
    response = model_gen.generate_content(prompt_parts)
    return response.text, retrieved_image_paths, text_results['documents'][0]

# --- App State Initialization ---


def initialize_session_state():
    """Initializes all session state variables if they don't exist."""
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "text_collection" not in st.session_state:
        st.session_state.text_collection = None
    if "image_collection" not in st.session_state:
        st.session_state.image_collection = None
    if "table_collection" not in st.session_state:
        st.session_state.table_collection = None
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = []
    if "tables" not in st.session_state:
        st.session_state.tables = []


def reset_app_state():
    """Clears all session state variables and the local images folder."""
    initialize_session_state()
    if os.path.exists("images"):
        shutil.rmtree("images")
    st.success("State cleared! Ready for a new document.")


initialize_session_state()

# --- UI Layout ---
st.title("🔬 AI Research Assistant")
st.markdown(
    "Upload a scientific paper (PDF) and ask questions about its text, images, and tables.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload your PDF", type=[
                                     "pdf"], on_change=reset_app_state)

    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing... This may take a moment."):
                raw_text, image_paths, tables = process_pdf_and_extract_content(
                    uploaded_file)
                client = chromadb.Client()

                if raw_text:
                    st.session_state.text_collection = embed_and_store_text(
                        chunk_text(raw_text), client)
                if image_paths:
                    st.session_state.image_collection = embed_and_store_images(
                        image_paths, client)
                if tables:
                    st.session_state.table_collection = embed_and_store_tables(
                        tables, client)

                st.session_state.image_paths, st.session_state.tables = image_paths, tables
                st.session_state.processed = True
            st.success("Document processed successfully!")
        else:
            st.warning("Please upload a PDF file first.")

    if st.session_state.processed:
        if st.button("Clear and Reset"):
            reset_app_state()

if not st.session_state.processed:
    st.info("Please upload and process a document to get started.")
else:
    st.header("Query Your Document")

    tab1, tab2, tab3 = st.tabs(
        ["💬 Q&A", "🖼️ Extracted Images", "📊 Extracted Tables"])

    with tab1:
        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            with st.spinner("Synthesizing answer..."):
                answer, image_sources, text_sources = get_final_answer(
                    user_question,
                    st.session_state.text_collection,
                    st.session_state.image_collection,
                    st.session_state.table_collection
                )
                st.write("**Answer:**")
                st.info(answer)

                with st.expander("Show Retrieved Context"):
                    st.write("**Text Chunks Used:**")
                    for i, text_chunk in enumerate(text_sources):
                        st.write(f"**Chunk {i+1}:**")
                        st.warning(text_chunk)

                    if image_sources:
                        st.write("**Image Sources Used:**")
                        st.image(image_sources, width=150)
    with tab2:
        if st.session_state.get("image_paths"):
            for img_path in st.session_state.image_paths:
                st.image(img_path, caption=os.path.basename(img_path))
        else:
            st.info("No images were found.")
    with tab3:
        if st.session_state.get("tables"):
            for i, table_md in enumerate(st.session_state.tables):
                st.write(f"**Table {i+1}**")
                st.markdown(table_md)
        else:
            st.info("No tables were found.")
