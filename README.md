# RAG-CHATBOT
RAG-CHATBOT

A Streamlit-based Retrieval-Augmented Generation assistant with document upload, local embeddings, user login, and per-user chat history.

🚀 Overview

RAG-CHATBOT is a fully local, privacy-friendly Retrieval-Augmented Generation (RAG) system built using Streamlit, ChromaDB, and Groq LLaMA Models (or any open-source LLM).
It allows users to:

Upload documents (PDF, TXT)

Generate embeddings locally

Chat with an AI assistant grounded on the uploaded knowledge

Maintain separate chat history per user

Authenticate using a secure login system

This project is ideal for students, researchers, enterprise users, and anyone building production-grade RAG applications.

✨ Features
🔐 User Authentication

Email + password login

Secure SHA-256 password hashing

User-specific session handling

💬 Chat History (Per User)

Each user sees only their own chat history

Messages stored in SQLite database

Persists across reloads

📄 Document Upload

Upload PDFs/TXT files

Automatic text extraction

Chunking & embedding generation

🧠 Local Vector Search

Uses ChromaDB for embeddings

Fast semantic search

No external dependencies required

🤖 RAG Pipeline

Retrieves top relevant chunks

Passes context → LLM

Produces accurate, grounded answers

🖥️ Streamlit UI

Clean modern interface

Chat-style conversation

Supports real-time retrieval & generation

🗂 Project Structure
RAG-CHATBOT/
│
├── app.py                # Streamlit main application (UI + Chat)
├── ingest.py             # Document ingestion + embedding creation
├── db.py                 # Chat history database (SQLite)
├── auth.py               # User login + signup system
├── config.py             # Model/API configuration
│
├── docs/chroma/          # Local ChromaDB vector store
│
├── README.md             # Project documentation
├── pyproject.toml        # Project dependencies
└── uv.lock               # Dependency lock file

🔧 Installation
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/RAG-CHATBOT.git
cd RAG-CHATBOT

2️⃣ Install dependencies

Using uv or pip:

pip install -r requirements.txt


or

uv sync

3️⃣ Run ingestion (optional)
python ingest.py

4️⃣ Start the Streamlit app
streamlit run app.py

🧩 How It Works
1. Upload Documents

Users upload PDF/TXT files → text is extracted → chunked → embeddings generated.

2. Store Embeddings Locally

ChromaDB stores embeddings in docs/chroma/.

3. Retrieve Relevant Chunks

User query → semantic search → top matching chunks returned.

4. Generate Answer

Query + retrieved context → sent to LLM → final grounded answer displayed.

🔐 Authentication & User Data

User credentials stored securely (hashed, not plaintext)

Chat history separated by user email

SQLite ensures fast and local storage

No data sent to third-party servers (unless you use cloud LLMs)

🤖 Models Used

You can use:

Groq Models (Recommended)

LLaMA-3 8B

Mixtral

Gemma

Or local models such as:

LLaMA-3 (GGUF)

Mistral

Qwen

🧪 Example Query

User: "Summarize the key points from the uploaded PDF."
Bot: Provides a context-grounded summary using retrieved chunks.

📌 Future Enhancements

Admin dashboard

JWT-based authentication

Multiple document workspaces per user

Vector store migrations

Response citations

PDF preview UI

🧑‍💻 Contributing

Contributions are welcome!
Feel free to submit issues or pull requests.

📜 License

This project is released under the MIT License.

⭐ Show Your Support

If you found this project useful, please star the repository on GitHub ❤️
