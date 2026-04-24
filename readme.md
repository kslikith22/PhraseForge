# PhraseForge — Plagiarism Detector & Remover

A fully local plagiarism detection and removal tool powered by Hugging Face Transformers. No external APIs, no data leaves your machine.

---

## How It Works

| Task | Model | Size |
|---|---|---|
| Plagiarism scoring | `distilgpt2` (perplexity-based) | ~330 MB |
| Paraphrasing | `Vamsi/T5_Paraphrase_Paws` (T5-small) | ~240 MB |

**Plagiarism scoring** uses GPT-2 perplexity — text that is highly predictable (low perplexity) matches patterns the model has memorised from common web text, indicating it may be copied. Higher perplexity means more original phrasing.

**Paraphrasing** uses a T5-small model fine-tuned on paraphrase pairs. Text is split into 150-word chunks, each chunk is paraphrased in two passes for maximum divergence, and all chunks are processed in parallel (4 threads) for speed.

---

## Requirements

- Python 3.11
- macOS / Linux / Windows
- ~600 MB disk space for models (downloaded automatically on first run)

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/phraseforge.git
cd phraseforge

# 2. Create conda environment
conda create -n plagiarism-remover python=3.11 -y
conda activate plagiarism-remover

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

Then open **http://localhost:8000** in your browser.

---

## Usage

1. Paste or type any amount of text into the input box (no length limit)
2. Click **Analyse** to see the plagiarism score, originality score, word count and risk level
3. Click **Remove Plagiarism** to paraphrase the text
4. Copy or download the output

---

## API Endpoints

The app also exposes a REST API. Interactive docs available at **http://localhost:8000/docs**.

### `POST /analyze`
Returns plagiarism score for the given text.
```json
// Request
{ "text": "Your text here..." }

// Response
{
  "plagiarism_score": 84.2,
  "originality_score": 15.8,
  "perplexity": 18.3,
  "risk_label": "High Risk",
  "word_count": 120,
  "sentence_count": 6,
  "char_count": 730
}
```

### `POST /paraphrase`
Paraphrases the text and returns a new plagiarism score.
```json
// Request
{ "text": "Your text here..." }

// Response
{
  "paraphrased_text": "...",
  "new_plagiarism_score": 61.0,
  "new_originality_score": 39.0,
  "new_risk_label": "Medium Risk",
  "new_perplexity": 38.7
}
```

---

## Performance (CPU)

| Word Count | Chunks | Estimated Time |
|---|---|---|
| 500 words | ~4 | ~2–3 min |
| 1000 words | ~7 | ~4–6 min |
| 5000 words | ~34 | ~8–10 min |

Times are for CPU (parallel, 4 threads). A GPU will be used automatically if available.

---

## Project Structure

```
phraseforge/
├── app.py            # FastAPI server + embedded HTML frontend
├── requirements.txt  # Python dependencies
└── README.md
```

---

## License

MIT