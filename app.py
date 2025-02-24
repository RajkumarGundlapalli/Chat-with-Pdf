import streamlit as st
from dotenv import load_dotenv
import pickle
import os
import google.generativeai as genai  # Gemini API
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Embeddings Model

# Load environment variables (Make sure you set GOOGLE_API_KEY in your .env file)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Google API Key is missing! Set it in your environment variables.")

# Load HuggingFace Embeddings (No API Key Needed)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar UI
with st.sidebar:
    st.title('LLM CHAT APP')
    st.markdown(''' 
            Welcome to the LLM Chat App  
            This app is an LLM-Powered Chatbot built using:  
            - [Streamlit](https://streamlit.io/)  
            - [LangChain](https://python.langchain.com/)  
            - Google Gemini API  
            ''')
    st.write('Made with ❤️ by Rajkumar')

def main():
    st.header('Chat with PDF 💭')

    # Upload PDF file
    pdf = st.file_uploader('Upload your PDF file', type=['pdf'])
    
    if pdf is not None:
        st.write(f"Uploaded file: {pdf.name}")

        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()

        if not text.strip():
            st.error("Could not extract text from the PDF. Please upload a valid text-based PDF.")
            return

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Store in FAISS vector database
        store_name = pdf.name.replace(".pdf", "").replace(" ", "_")
        faiss_index_file = f"{store_name}.pkl"

        if os.path.exists(faiss_index_file):
            with open(faiss_index_file, 'rb') as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loaded from existing FAISS index.")
        else:
            # Use correct embedding model instance
            VectorStore = FAISS.from_texts(chunks, embedding=embedding_model)

            # Save FAISS index
            with open(faiss_index_file, 'wb') as f:
                pickle.dump(VectorStore, f)
            st.write("New FAISS embeddings created and saved.")

        # Accept user query
        query = st.text_input("Ask me anything about this PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            if not docs:
                st.warning("No relevant information found in the PDF for this query.")
                return

            # Prepare context for Gemini API
            context = "\n\n".join([doc.page_content for doc in docs])

            try:
                model = genai.GenerativeModel("models/gemini-pro")  # ✅ Corrected Model Call
                response = model.generate_content(
                    f"Based on the following document excerpts, answer the question:\n\n{context}\n\nQuestion: {query}"
                )

                # Display Response
                st.write("### 🤖 AI Response:")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error in Gemini API request: {e}")

    else:
        st.warning("Please upload a PDF file.")

if __name__ == '__main__':
    main()
