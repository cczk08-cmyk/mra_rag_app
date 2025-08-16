import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

# --- Hardcoded login credentials ---
USERNAME = "admin"
PASSWORD = "password123"

# --- OpenAI API key ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_DB_PATH = "db"

# --- Session state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Login page ---
if not st.session_state.logged_in:
    st.title("Login Page")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pw == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful! Use the sidebar to navigate.")
        else:
            st.error("Invalid credentials")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Us", "Methodology"])

# --- Load PDF and TXT documents ---
def load_documents():
    texts = []

    # PDF
    pdf_path = "mra_supportable_activities.pdf"
    if os.path.exists(pdf_path):
        pdf_reader = PdfReader(pdf_path)
        pdf_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        if pdf_text.strip():
            texts.append(pdf_text)
            st.sidebar.success(f"Loaded PDF: {pdf_path}, {len(pdf_text)} chars")
        else:
            st.sidebar.warning(f"{pdf_path} contains no text.")
    else:
        st.sidebar.warning(f"{pdf_path} not found.")

    # TXT documents
    txt_files = ["mra_website.txt", "mra_faq.txt"]
    for txt_path in txt_files:
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                txt_text = f.read()
            if txt_text.strip():
                texts.append(txt_text)
                st.sidebar.success(f"Loaded TXT: {txt_path}, {len(txt_text)} chars")
            else:
                st.sidebar.warning(f"{txt_path} contains no text.")
        else:
            st.sidebar.warning(f"{txt_path} not found.")

    return texts

# --- Section-aware chunking ---
def chunk_by_section(text):
    sections = ["About MRA", "Eligibility", "How to Apply", "Supportable Activities", "FAQ"]
    chunks = []
    current_section = None
    buffer = []

    lines = text.split("\n")
    for line in lines:
        line_strip = line.strip()
        for sec in sections:
            if line_strip.lower().startswith(sec.lower()):
                if buffer:
                    chunks.append({"section": current_section, "text": " ".join(buffer)})
                    buffer = []
                current_section = sec
        if line_strip:
            buffer.append(line_strip)

    if buffer:
        chunks.append({"section": current_section, "text": " ".join(buffer)})

    # Split into smaller chunks
    final_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    for c in chunks:
        for sub_chunk in splitter.split_text(c["text"]):
            final_chunks.append({"section": c["section"], "text": sub_chunk})

    return final_chunks

# --- Cached vector db ---
@st.cache(allow_output_mutation=True)
def load_or_create_vector_db(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    all_chunks = []
    metadata = []

    for t in texts:
        section_chunks = chunk_by_section(t)
        for c in section_chunks:
            all_chunks.append(c["text"])
            metadata.append({"section": c["section"]})

    vectordb = Chroma.from_texts(all_chunks, embedding=embeddings,
                                 persist_directory=VECTOR_DB_PATH, metadatas=metadata)
    vectordb.persist()
    return vectordb, len(all_chunks)

# --- Load docs and create vector db ---
documents = load_documents()
if documents:
    vectordb, total_chunks = load_or_create_vector_db(documents)
    st.sidebar.success(f"Total chunks created: {total_chunks}")
else:
    vectordb = None
    st.sidebar.warning("No documents loaded. Vector DB not created.")

# --- Home Page ---
if page == "Home":
    st.title("MRA Grant Chatbot")
    st.write("Describe your overseas activity below and get guidance on whether it may be eligible for the MRA grant.")

    if vectordb is None:
        st.warning("No documents loaded. Chatbot cannot answer questions.")
    else:
        chat_container = st.container()
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_area("Your message:", height=100, placeholder="Type here...")
            send = st.form_submit_button("Send")
            restart = st.form_submit_button("Restart Chat")

        if restart:
            st.session_state.chat_history = []

        if send and query:
            st.session_state.chat_history.append({"role": "user", "content": query})

            # --- Retrieve top relevant chunks ---
            def retrieve_top_k(query, k=3):
                results = vectordb.similarity_search(query, k=k)
                return [{"section": r.metadata.get("section", "Unknown"), "text": r.page_content} for r in results]

            top_chunks = retrieve_top_k(query, k=3)
            context_text = "\n\n".join([f"[{c['section']}] {c['text']}" for c in top_chunks])

            # --- Build prompt ---
            prompt_text = f"""
You are an expert on Singapore's Market Readiness Assistance (MRA) Grant. 
Answer the user’s question based ONLY on the text below.
If the answer is not in the text, respond with "I don't know".

Context:
{context_text}

User question: {query}
"""
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
            ai_message = llm([HumanMessage(content=prompt_text)])
            answer = ai_message.content.strip()

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # --- Display chat ---
        for h in st.session_state.chat_history:
            if h["role"] == "user":
                chat_container.markdown(
                    f"<div style='color:#ffffff; background-color:#0b3d91; padding:8px; border-radius:5px; margin:5px 0'><b>You:</b> {h['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                chat_container.markdown(
                    f"<div style='color:#000000; background-color:#b2d3f5; padding:8px; border-radius:5px; margin:5px 0'><b>Assistant:</b> {h['content']}</div>",
                    unsafe_allow_html=True,
                )

# --- About Us ---
elif page == "About Us":
    st.title("About Us")
    st.write("""
Many companies exploring overseas expansion are often **uncertain whether their activities are eligible for grant support** under the Market Readiness Assistance (MRA) scheme.  
This chatbot enables companies to **self-serve for straightforward enquiries**, providing quick answers about the MRA grant without needing hotline assistance.  
In doing so, it **reduces workload on call centres and SME centres**, helping companies make informed decisions efficiently.

### Project Scope
Supports SMEs exploring overseas expansion and navigating MRA grant eligibility.

### Objectives
- Self-serve for common MRA questions
- Reduce SME centre workload
- Provide accurate, RAG-powered guidance

### Data Sources
- MRA PDFs, website text, FAQs

### Features
- Chatbot answering eligibility & supportable activity questions
- Section-aware RAG with retrieval from chunked documents
- Insight dashboard for SMEs
""")

# --- Methodology ---
elif page == "Methodology":
    st.title("Methodology")
    st.write("""
### Data Flows & Implementation
1. Load PDFs & TXT documents
2. Section-aware chunking (700 chars, 100-char overlap)
3. Convert chunks to embeddings & store in Chroma vector DB
4. Retrieve top-k chunks using cosine similarity
5. Generate answer using LLM with retrieved context

### Use Cases
1. Chat with Information
```mermaid
flowchart LR
    A[User Input] --> B[Retrieve top-k chunks from Vector DB]
    B --> C[Send chunks + query to LLM]
    C --> D[LLM generates answer]
    D --> E[Display answer to user])
    """)
