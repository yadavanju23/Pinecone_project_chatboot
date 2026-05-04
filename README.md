# 🤖 AI  Chatbot (RAG + LangChain + Flask)

A conversational AI chatbot built using **LangChain, Pinecone, and Groq**, capable of answering user queries based on custom PDF data using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

* 💬 Chat-based UI (HTML, CSS, JS)
* 🧠 Context-aware responses using memory
* 📄 PDF document ingestion
* 🔍 Semantic search with Pinecone
* ⚡ Fast LLM responses using Groq (LLaMA 3)
* 🌐 Flask backend integration

---

## 🏗️ Tech Stack

* **Backend:** Flask
* **LLM:** Groq (LLaMA 3.1)
* **Vector DB:** Pinecone
* **Embeddings:** HuggingFace (MiniLM)
* **Framework:** LangChain
* **Frontend:** HTML, CSS, JavaScript

---

## 📁 Project Structure

```
Pinecone_project_chatboot/
│
├── app.py
├── .env
├── requirements.txt
│
├── templates/
│   └── chatboot.html
│
├── static/
│   └── style.css
│
├── src/
│   ├── helper.py
│   └── prompt.py
│
└── data/
    └── (PDF files)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo.git
cd Pinecone_project_chatboot
```

---

### 2. Create virtual environment

```
conda create -n chatbot python=3.10
conda activate chatbot
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Setup environment variables

Create a `.env` file:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

---

### 5. Add your PDF data

Place your documents inside:

```
data/
```

---

### 6. Run the application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:8080
```

---

## 💡 How It Works

1. PDFs are loaded and split into chunks
2. Embeddings are created using HuggingFace
3. Stored in Pinecone vector database
4. User query → semantic search → relevant chunks
5. LLM generates contextual answer

---

## 🔗 API Endpoint

### POST `/get`

**Request:**

```
msg=your question
```

**Response:**

```
AI-generated answer
```

---

## ⚠️ Notes

* Make sure Pinecone index is created before running
* API keys must be valid
* Do not commit `.env` file

---

## 🚀 Deployment Options

* AWS EC2 (recommended)
* Render / Railway
* AWS Elastic Beanstalk

---

## 📌 Future Improvements

* Chat memory (multi-user sessions)
* Streaming responses
* Voice input/output
* Authentication system

---

## 👨‍💻 Author

**Anju Yadav**
AI & Web Developer

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
