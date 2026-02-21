# 🔍 LLM Document Analyzer

An AI-powered document analysis web app built with **Streamlit** and **OpenAI GPT-3.5-turbo**.  
Upload PDFs, Word docs, Excel files, CSVs, and plain text — then ask the LLM to summarize, extract entities, answer questions, compare documents, or run any custom query.

---

## ✨ Features

| Capability | Details |
|---|---|
| **Multi-format upload** | PDF, DOCX/DOC, XLSX/XLS, CSV, TXT, MD |
| **Multiple files** | Upload and analyze several docs at once |
| **Content preview** | See extracted text before analysis |
| **Summarize** | 3–5 paragraph summary |
| **Extract Entities** | People, orgs, dates, locations, money |
| **Q&A** | Ask any question; LLM answers from document |
| **Sentiment Analysis** | Tone & sentiment breakdown |
| **Compare Documents** | Similarities & differences across multiple files |
| **Custom Query** | Free-form prompt against the document |
| **In-memory processing** | Files are never written to disk |

---

## 🚀 Quick Start (local)

### 1. Clone & set up environment

```bash
git clone https://github.com/YOUR_USERNAME/llm-document-analyzer.git
cd llm-document-analyzer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set your OpenAI API key

**Option A — environment variable (simplest):**
```bash
export OPENAI_API_KEY="sk-..."     # Windows: set OPENAI_API_KEY=sk-...
```

**Option B — Streamlit secrets file:**
```bash
mkdir -p .streamlit
echo 'OPENAI_API_KEY = "sk-..."' > .streamlit/secrets.toml
```

### 3. Run

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Cloud (free, public URL)

1. **Fork** this repository on GitHub (button top-right of the repo page).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your fork → set **Main file path** to `main.py` → click **Deploy**.
4. Once deployed, open **Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Save** — the app will reboot with your key loaded. Share the generated `*.streamlit.app` URL with anyone.

---

## 🤗 Alternative: Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) → choose **Streamlit** SDK.
2. Push this repository to the Space (via the HF git remote or the web UI).
3. Add your key as a Space **Secret** (Settings → Variables and secrets):
   - Name: `OPENAI_API_KEY`  Value: `sk-...`
4. The Space will build automatically. Access it at `https://huggingface.co/spaces/YOUR_USERNAME/llm-document-analyzer`.

---

## 🗂️ Repository Structure

```
llm-document-analyzer/
├── main.py              # Streamlit app (all logic)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🔒 Security Notes

- API keys are loaded via Streamlit Secrets or environment variables — never hard-coded.
- Uploaded files are processed entirely in memory (`io.BytesIO`); nothing is written to disk.
- Content sent to OpenAI is truncated to ~12 000 characters (~3 k tokens) to cap cost and latency.

---

## 🛠️ Customization Tips

| Goal | Where to change |
|---|---|
| Use GPT-4 | Change `model="gpt-3.5-turbo"` in `call_llm()` |
| Raise content limit | Change `MAX_CONTENT_CHARS` in `main.py` |
| Add more file types | Extend `extract_content()` |
| Use a different LLM provider | Replace `call_llm()` with your provider's SDK |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `openai` | LLM API client |
| `pdfplumber` | PDF text & table extraction |
| `PyPDF2` | PDF fallback parser |
| `python-docx` | Word document parsing |
| `pandas` | Excel / CSV parsing |
| `openpyxl` | Excel engine for pandas |

---

## 📝 License

MIT — free to use, modify, and distribute.
