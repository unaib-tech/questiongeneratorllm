# import os
# import io
# import hashlib
# import pdfplumber
# import tempfile
# import streamlit as st
# import csv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.llms import Ollama
# from dotenv import load_dotenv
# import google.generativeai as genai
# from concurrent.futures import ThreadPoolExecutor
# from langchain_core.prompts import PromptTemplate

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Constants
# CHUNK_SIZE = 3000  # Optimal for modern LLM context windows
# CHUNK_OVERLAP = 300

# # Helper Functions
# def extract_text_from_pdf(pdf_path):
#     """Extract text from PDF using pdfplumber with error handling."""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             return "\n".join(page.extract_text() or "" for page in pdf.pages)
#     except Exception as e:
#         st.error(f"PDF extraction error: {str(e)}")
#         return ""

# def calculate_file_hash(file_content):
#     """Calculate SHA256 hash of file content."""
#     return hashlib.sha256(file_content).hexdigest()

# def process_pdf(file_content):
#     """Process PDF content with temporary file handling."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(file_content)
#         temp_file_path = temp_file.name
    
#     try:
#         text = extract_text_from_pdf(temp_file_path)
#     finally:
#         os.unlink(temp_file_path)
#     return text

# def generate_questions(section, llm):
#     """Generate questions for a text chunk using structured prompt."""
#     prompt_template = PromptTemplate.from_template(
#         "Generate concise questions about key concepts in this text. "
#         "Prioritize questions that test deep understanding. "
#         "Format each question on a new line.\n\nText: {context}"
#     )
#     formatted_prompt = prompt_template.format(context=section)
    
#     try:
#         if isinstance(llm, ChatGoogleGenerativeAI):
#             response = llm.invoke(formatted_prompt).content
#         else:
#             response = llm.invoke(formatted_prompt)
#         return [q.strip() for q in response.split("\n") if q.strip()]
#     except Exception as e:
#         st.error(f"Generation error: {str(e)}")
#         return []

# # Streamlit App
# def main():
#     st.set_page_config(page_title="PDF Question Generator", page_icon="📚")
#     st.title("Smart PDF Question Generator")
    
#     # Sidebar Controls
#     with st.sidebar:
#         st.header("Configuration")
#         model_choice = st.selectbox("LLM Provider", ["Gemini", "Mistral"])
#         uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
#     # Initialize LLM only once
#     if "llm" not in st.session_state:
#         if model_choice == "Gemini":
#             st.session_state.llm = ChatGoogleGenerativeAI(
#                 model="gemini-1.5-flash",
#                 temperature=0.4,
#                 model_kwargs={"top_p": 0.7}
#             )
#         else:
#             st.session_state.llm = Ollama(model="mistral", temperature=0.5)
    
#     # File Processing
#     if uploaded_files:
#         unique_files = {}
#         with st.status("Processing PDFs...", expanded=True) as status:
#             st.write("Validating files...")
#             for file in uploaded_files:
#                 file_content = file.getvalue()
#                 file_hash = calculate_file_hash(file_content)
#                 if file_hash not in unique_files:
#                     unique_files[file_hash] = file_content
            
#             st.write(f"Processing {len(unique_files)} unique files...")
#             with ThreadPoolExecutor() as executor:
#                 futures = [executor.submit(process_pdf, content) for content in unique_files.values()]
#                 texts = [future.result() for future in futures]
            
#             combined_text = "\n\n".join(filter(None, texts))
#             status.update(label="Processing complete!", state="complete", expanded=False)
        
#         # Question Generation
#         if st.button("Generate Questions"):
#             with st.spinner("Analyzing content and generating questions..."):
#                 splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=CHUNK_SIZE,
#                     chunk_overlap=CHUNK_OVERLAP
#                 )
#                 chunks = splitter.split_text(combined_text)
                
#                 with ThreadPoolExecutor(max_workers=4) as executor:
#                     questions = []
#                     futures = [executor.submit(generate_questions, chunk, st.session_state.llm) for chunk in chunks]
#                     for future in futures:
#                         questions.extend(future.result())
                    
#                     # Deduplicate while preserving order
#                     seen = set()
#                     unique_questions = [q for q in questions if not (q in seen or seen.add(q))]
            
#             # Display Results
#             st.subheader(f"Generated Questions ({len(unique_questions)})")
#             with st.expander("View All Questions", expanded=True):
#                 for i, q in enumerate(unique_questions, 1):
#                     st.markdown(f"{i}. {q}")
            
#             # CSV Export
#             if unique_questions:
#                 # Create in-memory CSV file
#                 csv_buffer = io.BytesIO()
#                 csv_writer = csv.writer(csv_buffer)
#                 csv_writer.writerow(["Question"])
#                 csv_writer.writerows([[q] for q in unique_questions])
#                 csv_buffer.seek(0)  # Reset buffer position
                
#                 st.download_button(
#                     label="Download Questions",
#                     data=csv_buffer.getvalue(),
#                     file_name="generated_questions.csv",
#                     mime="text/csv"
#                 )
import os
import io
import hashlib
import pdfplumber
import tempfile
import streamlit as st
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
CHUNK_SIZE = 3000  # Optimal for modern LLM context windows
CHUNK_OVERLAP = 300

# Helper Functions
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber with error handling."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

def calculate_file_hash(file_content):
    """Calculate SHA256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()

def process_pdf(file_content):
    """Process PDF content with temporary file handling."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    try:
        text = extract_text_from_pdf(temp_file_path)
    finally:
        os.unlink(temp_file_path)
    return text

def generate_questions(section, llm):
    """Generate questions for a text chunk using structured prompt."""
    prompt_template = PromptTemplate.from_template(
        "Generate concise questions about key concepts in this text. "
        "Prioritize questions that test deep understanding. "
        "Format each question on a new line.\n\nText: {context}"
    )
    formatted_prompt = prompt_template.format(context=section)
    
    try:
        if isinstance(llm, ChatGoogleGenerativeAI):
            response = llm.invoke(formatted_prompt).content
        else:
            response = llm.invoke(formatted_prompt)
        return [q.strip() for q in response.split("\n") if q.strip()]
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return []

# Streamlit App
def main():
    st.set_page_config(page_title="PDF Question Generator", page_icon="📚")
    st.title("Smart PDF Question Generator")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("LLM Provider", ["Gemini", "Mistral"])
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Initialize LLM only once
    if "llm" not in st.session_state:
        if model_choice == "Gemini":
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.4,
                model_kwargs={"top_p": 0.7}
            )
        else:
            st.session_state.llm = Ollama(model="mistral", temperature=0.5)
    
    # File Processing
    if uploaded_files:
        unique_files = {}
        with st.spinner("Processing PDFs..."):
            st.write("Validating files...")
            for file in uploaded_files:
                file_content = file.getvalue()
                file_hash = calculate_file_hash(file_content)
                if file_hash not in unique_files:
                    unique_files[file_hash] = file_content
            
            st.write(f"Processing {len(unique_files)} unique files...")
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_pdf, content) for content in unique_files.values()]
                texts = [future.result() for future in futures]
            
            combined_text = "\n\n".join(filter(None, texts))
        
        # Question Generation
        if st.button("Generate Questions"):
            with st.spinner("Analyzing content and generating questions..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_text(combined_text)
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    questions = []
                    futures = [executor.submit(generate_questions, chunk, st.session_state.llm) for chunk in chunks]
                    for future in futures:
                        questions.extend(future.result())
                    
                    # Deduplicate while preserving order
                    seen = set()
                    unique_questions = [q for q in questions if not (q in seen or seen.add(q))]
            
            # Display Results
            st.subheader(f"Generated Questions ({len(unique_questions)})")
            with st.expander("View All Questions", expanded=True):
                for i, q in enumerate(unique_questions, 1):
                    st.markdown(f"{i}. {q}")
            
            # CSV Export
            if unique_questions:
                # Create in-memory CSV file
                csv_buffer = io.BytesIO()
                csv_writer = csv.writer(csv_buffer)
                csv_writer.writerow(["Question"])
                csv_writer.writerows([[q] for q in unique_questions])
                csv_buffer.seek(0)  # Reset buffer position
                
                st.download_button(
                    label="Download Questions",
                    data=csv_buffer.getvalue(),
                    file_name="generated_questions.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
