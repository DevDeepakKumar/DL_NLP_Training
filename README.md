# RAG from Scratch — 10-K Document Q&A

A beginner-friendly Jupyter notebook that builds a complete **Retrieval-Augmented Generation (RAG)** pipeline over a financial 10-K PDF report — from raw PDF text to a grounded LLM answer.

Built as a step-by-step teaching notebook: every code cell has a markdown explanation above it.

## What it does

1. **Ingest** — reads the PDF page by page (`pdfplumber`), embeds each page (`sentence-transformers`), and stores the embeddings in a local vector database (`chromadb`).
2. **Retrieve** — embeds the user's question and finds the most similar pages in the vector database.
3. **Generate** — sends the retrieved pages + question to an LLM on **Groq** (`llama-3.3-70b-versatile`), which answers using only that context.

## Tech stack

| Purpose | Library |
|---|---|
| PDF text extraction | `pdfplumber` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database | `chromadb` |
| LLM inference | `groq` |
| Environment variables | `python-dotenv` |

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install pdfplumber sentence-transformers chromadb groq python-dotenv
```
(Or run the `!pip install ...` cell at the top of the notebook if you're using Google Colab.)

### 3. Add your PDF
Place your 10-K (or any) PDF in the project folder and update the `PDF_PATH` variable in the notebook if the filename differs from `_10-K-2025-As-Filed.pdf`.

### 4. Get a Groq API key

Groq provides free, very fast inference for open models like Llama 3.3.

1. Go to **[console.groq.com](https://console.groq.com)** and sign up / log in (you can use a Google account).
2. In the left sidebar, click **API Keys**.
3. Click **Create API Key**, give it a name (e.g. `rag-notebook`), and click **Submit**.
4. Copy the key immediately — Groq only shows it once.

### 5. Store the key in a `.env` file
Create a file named `.env` in the project root:
```
GROQ_API_KEY=your_key_here
```
> ⚠️ Never commit your `.env` file. Add it to `.gitignore`.

### 6. Run the notebook
Open `RAG_from_scratch_10K.ipynb` in Jupyter or Google Colab and run the cells in order, top to bottom.

## Project structure
```
.
├── RAG_from_scratch_10K.ipynb   # main notebook
├── _10-K-2025-As-Filed.pdf      # your source PDF (not included)
├── .env                          # your Groq API key (not committed)
├── my_chromadb/                  # local vector DB (auto-created on first run)
└── README.md
```

## Notes
- The vector database persists on disk in `./my_chromadb`, so you don't need to re-ingest the PDF every time — only when the PDF changes.
- Try changing `top_k` in `retrieve_docs()` to control how many pages are retrieved as context.
- Swap in a different PDF by changing `PDF_PATH` — no other code changes needed.

## License
MIT (or update to your preferred license).
