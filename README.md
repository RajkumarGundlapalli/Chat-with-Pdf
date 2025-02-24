# LLM Chat App

This is a PDF-based chatbot powered by Large Language Models (LLMs). The app allows users to upload a PDF and ask questions about its content. It uses Streamlit for the interface, LangChain for text processing, and Google Gemini API for generating responses.

## Features
- Upload a PDF and extract its text content
- Chunk the text and create embeddings using HuggingFace models
- Store and retrieve embeddings using FAISS vector database
- Use Google Gemini API to answer user queries based on PDF content

## Technologies Used
- [Streamlit](https://streamlit.io/): Web app framework for Python
- [LangChain](https://python.langchain.com/): Library for LLM-powered applications
- [Google Gemini API](https://ai.google.dev/): API for generative AI
- [HuggingFace Embeddings](https://huggingface.co/): Pretrained language models
- [FAISS](https://faiss.ai/): Vector search library

## Prerequisites
1. Python 3.8+
2. Google Gemini API Key (set in `.env` file)

## Setup
1. Clone this repository:
```bash
git clone https://github.com/RajkumarGundlapalli/Chat-with-Pdf.git
cd Chat-with-Pdf
```
2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file and add your Google API key:
```env
GOOGLE_API_KEY=your_google_api_key
```

## How to Run
```bash
streamlit run app.py
```

## Usage
1. Upload a PDF file.
2. Ask a question related to the PDF content.
3. Get
4.  an AI-generated response based on the document.

## Output 
This is the user interface of this Application
![image](https://github.com/user-attachments/assets/dff6985b-74e5-4811-99c2-48a6c52112e1)




This the Output of the model when a question is asked after uploading the pdf file 
![image](https://github.com/user-attachments/assets/3b024505-075a-46fd-b4c9-528583e0068a)
