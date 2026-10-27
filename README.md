# RAG over PDFs

A small retrieval-augmented generation pipeline: load PDFs, split them into chunks,
embed the chunks, store them in a local Chroma collection, retrieve the closest
chunks for a question, and have a Groq-hosted LLM answer **from those chunks only**.

## Layout

```
data/pdf/                 the source documents (Java.pdf, Python.pdf)
notebook/                 the walkthrough
  pdf_loader.ipynb        ingestion -> embeddings -> vector store -> retrieval -> answer
  document.ipynb          loading text and PDF documents with LangChain loaders
src/rag/retrieval.py      the same pipeline as an importable module
app.py                    Streamlit front end (answers + the sources it used)
tests/                    13 tests over the module and the notebook's own cells
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "pytest>=8"
```

Create a `.env` in the repo root with your Groq key. It is gitignored.

```
GROQ_API_KEY=your-key-here
```

## Run the tests

```bash
python -m pytest -v
```

No API key is needed: the LLM client is stubbed. The first run downloads the
`all-MiniLM-L6-v2` embedding model (~90 MB) from Hugging Face.

## Run the notebook

Open `notebook/pdf_loader.ipynb` and run the cells top to bottom. Cell 13 rebuilds
the vector store from `data/pdf`, so re-running never accumulates stale chunks.

## Run the app locally

```bash
streamlit run app.py
```

Then open the printed localhost URL.

## Hosting on Streamlit Community Cloud (free tier)

1. Push this repo to GitHub (already done for the upstream repo).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **New app**.
3. Pick the repository and branch, and set **Main file path** to `app.py`.
4. In the same panel, add the secrets (this is the deployed equivalent of `.env`):

   ```toml
   GROQ_API_KEY = "your-key-here"
   APP_PASSWORD = "something-only-you-share"
   # Optional, while you are getting the password right: adds the configured
   # password's length to the wrong-password message. Remove it afterwards.
   # APP_PASSWORD_DEBUG = "1"
   ```

   Use a **second, separate Groq key** here so you can revoke the public copy
   without touching the key you use locally. `APP_PASSWORD` puts the app behind a
   password, which is worth doing on a public URL because every question spends
   tokens on that key. If you leave `APP_PASSWORD` out, the app stays open but shows
   a warning banner saying so.
5. Deploy. First boot takes a minute or two: it downloads the embedding model and
   builds the index from the two PDFs.

There is no requirements-file field to set: Streamlit Community Cloud installs
`requirements.txt` from the repo root automatically, and that file already pins the
CPU-only torch wheel.

### If a deployed app reports a missing module

The traceback Streamlit shows on a deployed app is redacted. Open the
**Environment diagnostics** panel at the bottom of the app: it lists every module the
app needs, whether it imported, and its version, so a missing package names itself.
Also check that `requirements.txt` lists it, then reboot the app from **Manage app**.

Things worth knowing about the free tier:

- Apps sleep after 12 hours without traffic, and waking one needs a visitor to click
  through an interstitial. A cold start also re-downloads the embedding model.
- The free tier gives about 1 GB of memory. This app peaks around 633 MB with the CPU
  torch pin, so there is headroom but not a lot; re-check it if you add large PDFs.
- Every question spends Groq tokens on that key. `app.py` caps questions per session
  (`MAX_QUESTIONS_PER_SESSION`) and refuses to answer when retrieval finds nothing
  relevant, rather than letting the model answer from its own knowledge.

## How retrieval is scored

Chroma's default space is L2 and it returns a **squared** distance. The embeddings
are unit length, so `distance = 2 - 2*cos`, which makes the cosine similarity
`1 - distance/2`. The pipeline converts scores that way, so `score_threshold` and
`MIN_SCORE` are plain cosine values (0.2 is a reasonable floor). See
`cosine_similarity()` in `src/rag/retrieval.py` for the per-space formulas.
