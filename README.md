# CV Chatbot (RAG-based)

An intelligent chatbot that answers questions about a CV using a **Retrieval-Augmented Generation (RAG)** pipeline.

## Features

*  Upload CVs in **PDF or DOCX**
*  Automatic text cleaning & preprocessing
*  Smart text chunking
*  Semantic search using embeddings
*  Vector storage with FAISS
*  Answer generation using Gemini (Google Generative AI)
*  Security: answers only CV-related questions

---

## Architecture (RAG Pipeline)

1. **Load CV** → Extract text from PDF/DOCX
2. **Preprocess** → Clean text
3. **Chunking** → Split into smaller parts
4. **Embedding** → Convert text into vectors
5. **Indexing** → Store vectors in FAISS
6. **Retrieval** → Get relevant chunks
7. **Prompting** → Build controlled prompt
8. **Generation** → Generate answer (Gemini)
9. **Security** → Block out-of-context questions

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GoDD3st/Chatbot-RAG-sur-CV.git
cd Chatbot-RAG-sur-CV
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate it

**Windows (CMD):**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Add environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ How to Use

### Run the application:

```bash
streamlit run app.py
```

### Steps:

1. Upload a CV (PDF or DOCX)
2. Wait for processing
3. Ask questions like:

   * "What skills does the candidate have?"
   * "What is his experience?"
   * "Where did he study?"

---

##  Notes

* The chatbot only answers **based on the CV content**
* If the answer is not found, it will refuse to respond
* Make sure your API key is valid

---

## 📌 Technologies Used

* Python
* Streamlit
* FAISS
* HuggingFace Embeddings
* Google Gemini API

---

##  Future Improvements

* Add conversation memory
* Improve ranking (reranking)
* Deploy online (Streamlit Cloud)

---

##  Author
GoDD3st
