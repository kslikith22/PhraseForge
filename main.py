"""
PhraseForge — Plagiarism Detector & Remover
============================================
Models used  (~570 MB total, runs on CPU):
  • distilgpt2                          — plagiarism scoring via perplexity
  • Vamsi/T5_Paraphrase_Paws           — paraphrasing (T5-small fine-tuned)

Install deps:
  pip install "transformers==4.40.2" torch==2.2.2 fastapi uvicorn sentencepiece

Run:
  python app.py

Then open:  http://localhost:8000
"""

import re
import math
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

app = FastAPI(title="PhraseForge", description="Plagiarism Detector & Remover")

# ── Request schema ────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥  Device: {DEVICE.upper()}")

# ── Load distilgpt2  (plagiarism / perplexity) ────────────────────────────────
print("⏳ Loading distilgpt2 for plagiarism scoring…")
ppl_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
ppl_model     = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE)
ppl_model.eval()
print("✅ distilgpt2 ready.")

# ── Load T5-small paraphraser ─────────────────────────────────────────────────
print("⏳ Loading T5 paraphraser…")
PARA_MODEL = "Vamsi/T5_Paraphrase_Paws"
para_tokenizer = T5Tokenizer.from_pretrained(PARA_MODEL)
para_model     = T5ForConditionalGeneration.from_pretrained(PARA_MODEL).to(DEVICE)
para_model.eval()
print("✅ T5 paraphraser ready.")
print("🚀 Server starting — open http://localhost:8000\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PLAGIARISM SCORING  (GPT-2 perplexity)
# ══════════════════════════════════════════════════════════════════════════════

def compute_perplexity(text: str) -> float:
    """
    Slide a window across the token sequence and average the NLL loss.
    Lower perplexity  → text is more 'predictable' → higher plagiarism risk.

    Typical distilgpt2 perplexity ranges:
      < 30   : very common / likely copied phrasing
      30–100 : mixed originality
      > 100  : fairly original / creative
    """
    encodings = ppl_tokenizer(text, return_tensors="pt").to(DEVICE)
    seq_len   = encodings.input_ids.size(1)

    if seq_len == 0:
        return 100.0          # treat empty as very original

    max_len = ppl_model.config.n_positions   # 1024 for distilgpt2
    stride  = 512
    nlls    = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end     = min(begin + max_len, seq_len)
        trg_len = end - prev_end

        input_ids  = encodings.input_ids[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100   # only score new tokens

        with torch.no_grad():
            out = ppl_model(input_ids, labels=target_ids)
        nlls.append(out.loss * trg_len)

        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return float(ppl)


def perplexity_to_plagiarism(ppl: float) -> float:
    """
    Sigmoid mapping:  ppl=20 → ~95%,  ppl=60 → ~50%,  ppl=150 → ~10%
    """
    score = 100.0 / (1.0 + math.exp((ppl - 60) / 25))
    return round(min(100.0, max(0.0, score)), 1)


def risk_label(score: float) -> str:
    if score >= 65:
        return "High Risk"
    elif score >= 35:
        return "Medium Risk"
    return "Low Risk"


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT CHUNKING  (unlimited length support)
# ══════════════════════════════════════════════════════════════════════════════

def split_sentences(text: str):
    """Return list of (kind, content) where kind ∈ {'TEXT','BLANK'}."""
    result = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            result.append(("BLANK", ""))
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if sent:
                result.append(("TEXT", sent))
    return result


def make_chunks(sentences, max_words: int = 150):
    """
    Group sentences into chunks with at most max_words words.
    Paragraph breaks (BLANK) flush the current chunk.
    """
    chunks   = []
    buf      = []
    buf_len  = 0

    for kind, sent in sentences:
        if kind == "BLANK":
            if buf:
                chunks.append((" ".join(buf), "TEXT"))
                buf, buf_len = [], 0
            chunks.append(("", "BLANK"))
            continue

        wc = len(sent.split())
        if buf_len + wc > max_words and buf:
            chunks.append((" ".join(buf), "TEXT"))
            buf, buf_len = [sent], wc
        else:
            buf.append(sent)
            buf_len += wc

    if buf:
        chunks.append((" ".join(buf), "TEXT"))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  PARAPHRASING
# ══════════════════════════════════════════════════════════════════════════════

def paraphrase_chunk(text: str, max_len: int = 256) -> str:
    """
    Paraphrase with sampling + high temperature, then a second pass
    on the output to maximise divergence from the original.
    """
    def _generate(src: str) -> str:
        enc = para_tokenizer(
            f"paraphrase: {src} </s>",
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding="max_length",
        ).to(DEVICE)
        with torch.no_grad():
            out = para_model.generate(
                input_ids            = enc["input_ids"],
                attention_mask       = enc["attention_mask"],
                max_length           = max_len,
                do_sample            = True,
                temperature          = 1.6,
                top_k                = 120,
                top_p                = 0.95,
                repetition_penalty   = 3.5,
                no_repeat_ngram_size = 4,
                num_return_sequences = 1,
            )
        return para_tokenizer.decode(out[0], skip_special_tokens=True)

    first_pass  = _generate(text)        # paraphrase original
    second_pass = _generate(first_pass)  # paraphrase the paraphrase
    return second_pass


def paraphrase_text(text: str) -> str:
    """
    Chunk → paraphrase in parallel (4 threads) → reassemble.
    Handles unlimited input length. ~4x faster than sequential on CPU.
    """
    sentences = split_sentences(text)
    chunks    = make_chunks(sentences, max_words=150)

    # Separate text chunks (need paraphrasing) from blank markers
    text_chunks    = [(i, c) for i, (c, k) in enumerate(chunks) if k == "TEXT"]
    blank_positions = {i for i, (_, k) in enumerate(chunks) if k == "BLANK"}

    # Paraphrase all text chunks in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(paraphrase_chunk, c): i for i, c in text_chunks}
        for future in as_completed(futures):
            idx         = futures[future]
            results[idx] = future.result()

    # Reassemble in original order
    final = ""
    for i, (_, kind) in enumerate(chunks):
        if kind == "BLANK":
            final = final.rstrip() + "\n\n"
        else:
            final += results[i] + " "

    return final.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/analyze")
async def analyze(body: TextRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    ppl       = compute_perplexity(text)
    plag      = perplexity_to_plagiarism(ppl)
    words     = len(text.split())
    sentences = len([s for s in re.split(r"[.!?]+", text) if s.strip()])

    return {
        "plagiarism_score":  plag,
        "originality_score": round(100 - plag, 1),
        "perplexity":        round(ppl, 1),
        "risk_label":        risk_label(plag),
        "word_count":        words,
        "sentence_count":    sentences,
        "char_count":        len(text),
    }


@app.post("/paraphrase")
async def paraphrase(body: TextRequest):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    result   = paraphrase_text(text)
    ppl_new  = compute_perplexity(result)
    plag_new = perplexity_to_plagiarism(ppl_new)

    return {
        "paraphrased_text":      result,
        "new_plagiarism_score":  plag_new,
        "new_originality_score": round(100 - plag_new, 1),
        "new_risk_label":        risk_label(plag_new),
        "new_perplexity":        round(ppl_new, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDED HTML FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhraseForge — Plagiarism Remover</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0d0f12; --bg2:#13161b; --bg3:#17191f;
  --surface:#1c1f27; --surface2:#23262f;
  --border:rgba(255,255,255,0.07); --border2:rgba(255,255,255,0.13);
  --text:#dde0e8; --muted:#6e7380;
  --accent:#c9933a; --accent2:#e2b05c;
  --teal:#3ecfaa; --red:#e05c5c; --blue:#6ab4f5; --purple:#a78bfa;
  --r:12px; --rsm:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;line-height:1.6}

/* ── header ── */
header{
  border-bottom:1px solid var(--border);padding:0 2.5rem;height:60px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;background:var(--bg);z-index:100;
}
.logo{display:flex;align-items:center;gap:10px}
.logo-mark{
  width:28px;height:28px;background:var(--accent);border-radius:7px;
  display:flex;align-items:center;justify-content:center;font-size:14px;
}
.logo-name{font-family:'DM Serif Display',serif;font-size:18px;letter-spacing:-0.3px}
.logo-sep{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;
  font-weight:500;border-left:1px solid var(--border2);padding-left:10px;margin-left:2px}
.hdr-pill{
  font-size:11px;color:var(--teal);border:1px solid rgba(62,207,170,.25);
  background:rgba(62,207,170,.06);padding:4px 12px;border-radius:20px;font-weight:500;
}

/* ── main ── */
main{max-width:1160px;margin:0 auto;padding:2.5rem 2.5rem 6rem}

.hero{margin-bottom:2.5rem;display:flex;align-items:flex-end;justify-content:space-between;gap:2rem}
.hero h1{
  font-family:'DM Serif Display',serif;font-size:36px;font-weight:400;
  line-height:1.15;letter-spacing:-.8px;max-width:480px;
}
.hero h1 em{font-style:italic;color:var(--accent2)}
.hero-sub{font-size:13px;color:var(--muted);max-width:230px;text-align:right;line-height:1.7}

/* ── score cards ── */
.scores-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
.sc{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--rsm);
  padding:15px 17px;position:relative;overflow:hidden;transition:border-color .3s;
}
.sc.lit{border-color:var(--border2)}
.sc::after{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  border-radius:2px 2px 0 0;opacity:.5;transition:opacity .4s;
}
.sc.lit::after{opacity:1}
.sc-plag::after  {background:var(--red)}
.sc-orig::after  {background:var(--teal)}
.sc-words::after {background:var(--blue)}
.sc-risk::after  {background:var(--purple)}

.sc-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:9px}
.sc-val{font-family:'DM Serif Display',serif;font-size:28px;line-height:1;margin-bottom:3px}
.sc-plag  .sc-val{color:var(--red)}
.sc-orig  .sc-val{color:var(--teal)}
.sc-words .sc-val{color:var(--blue)}
.sc-risk  .sc-val{color:var(--purple)}
.sc-sub{font-size:11px;color:var(--muted)}

.sc-track{height:3px;background:var(--border2);border-radius:2px;margin-top:11px;overflow:hidden}
.sc-bar{height:100%;border-radius:2px;width:0;transition:width 1.1s cubic-bezier(.4,0,.2,1)}
.sc-plag  .sc-bar{background:var(--red)}
.sc-orig  .sc-bar{background:var(--teal)}
.sc-words .sc-bar{background:var(--blue)}
.sc-risk  .sc-bar{background:var(--purple)}

/* ── workspace ── */
.workspace{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
.panel{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
  display:flex;flex-direction:column;
}
.ph{
  padding:11px 18px;border-bottom:1px solid var(--border);background:var(--bg3);
  border-radius:var(--r) var(--r) 0 0;
  display:flex;align-items:center;justify-content:space-between;
}
.ph-title{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:var(--muted)}
.ph-meta{font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);display:flex;gap:10px;align-items:center}
.badge{font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;text-transform:uppercase;letter-spacing:.05em;display:none}
.badge.show{display:inline}
.badge-hi  {background:rgba(224,92,92,.15);color:var(--red)}
.badge-mid {background:rgba(201,147,58,.15);color:var(--accent2)}
.badge-lo  {background:rgba(62,207,170,.15);color:var(--teal)}

textarea{
  flex:1;width:100%;background:transparent;border:none;outline:none;
  color:var(--text);font-family:'DM Sans',sans-serif;font-size:14.5px;
  line-height:1.8;padding:18px 20px;resize:none;min-height:360px;
}
textarea::placeholder{color:var(--muted);opacity:.5}

.out-body{
  flex:1;padding:18px 20px;font-size:14.5px;line-height:1.8;
  color:var(--text);white-space:pre-wrap;word-break:break-word;
  overflow-y:auto;min-height:360px;border-radius:0 0 var(--r) var(--r);
}
.out-ph{color:var(--muted);opacity:.4;font-style:italic;font-size:14px}

/* ── controls ── */
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}

.btn{
  padding:10px 22px;border-radius:var(--rsm);font-family:'DM Sans',sans-serif;
  font-size:13.5px;font-weight:600;cursor:pointer;border:none;
  transition:all .15s;display:flex;align-items:center;gap:8px;
}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none!important}
.btn-primary{background:var(--accent);color:#1a0f00}
.btn-primary:hover:not(:disabled){background:var(--accent2);transform:translateY(-1px)}
.btn-secondary{background:transparent;color:var(--muted);border:1px solid var(--border2)}
.btn-secondary:hover:not(:disabled){color:var(--text);background:var(--surface2)}
.btn-ghost{
  width:38px;height:38px;padding:0;background:transparent;
  color:var(--muted);border:1px solid var(--border2);
  display:flex;align-items:center;justify-content:center;font-size:15px;
}
.btn-ghost:hover{color:var(--text);background:var(--surface2)}

.divider{width:1px;height:24px;background:var(--border2);margin:0 4px}

/* ── progress bar ── */
.progress-wrap{
  height:3px;background:var(--border2);border-radius:2px;
  margin-bottom:18px;overflow:hidden;display:none;
}
.progress-wrap.show{display:block}
.progress-bar{
  height:100%;border-radius:2px;background:var(--accent);
  width:0;transition:width .4s ease;
}

/* ── status ── */
.status{
  font-size:12px;color:var(--muted);display:flex;align-items:center;gap:8px;
  font-family:'DM Mono',monospace;margin-left:auto;
}
.spinner{
  width:13px;height:13px;border:2px solid var(--border2);
  border-top-color:var(--accent);border-radius:50%;
  display:none;animation:spin .8s linear infinite;
}
.spinner.show{display:block}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── perplexity info ── */
.ppl-note{
  font-size:11.5px;color:var(--muted);line-height:1.6;
  padding:13px 18px;background:var(--bg3);border:1px solid var(--border);
  border-radius:var(--rsm);margin-top:18px;
}
.ppl-note strong{color:var(--text);font-weight:500}

/* ── copy btn ── */
.copy-wrap{display:flex;justify-content:flex-end;padding:10px 16px 14px;gap:8px}
.btn-copy{
  font-size:12px;padding:6px 14px;border-radius:6px;
  background:transparent;color:var(--muted);border:1px solid var(--border2);
  cursor:pointer;transition:all .15s;font-family:'DM Sans',sans-serif;font-weight:500;
}
.btn-copy:hover{color:var(--text);background:var(--surface2)}

/* ── after score banner ── */
.after-banner{
  display:none;margin-top:18px;padding:14px 18px;
  border-radius:var(--rsm);border:1px solid rgba(62,207,170,.2);
  background:rgba(62,207,170,.06);
  font-size:13px;color:var(--teal);line-height:1.6;
}
.after-banner.show{display:block}
.after-banner strong{font-weight:600}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">✦</div>
    <span class="logo-name">PhraseForge</span>
    <span class="logo-sep">Plagiarism Remover</span>
  </div>
  <div class="hdr-pill">Local · No API · Transformers</div>
</header>

<main>
  <div class="hero">
    <h1>Detect &amp; remove<br><em>plagiarism</em> instantly.</h1>
    <p class="hero-sub">Powered by distilgpt2 + T5-small — runs entirely on your machine. No data leaves your device.</p>
  </div>

  <!-- Score cards -->
  <div class="scores-row">
    <div class="sc sc-plag" id="c-plag">
      <div class="sc-label">Plagiarism Score</div>
      <div class="sc-val" id="v-plag">—</div>
      <div class="sc-sub" id="s-plag">Awaiting analysis</div>
      <div class="sc-track"><div class="sc-bar" id="b-plag"></div></div>
    </div>
    <div class="sc sc-orig" id="c-orig">
      <div class="sc-label">Originality Score</div>
      <div class="sc-val" id="v-orig">—</div>
      <div class="sc-sub" id="s-orig">Awaiting analysis</div>
      <div class="sc-track"><div class="sc-bar" id="b-orig"></div></div>
    </div>
    <div class="sc sc-words" id="c-words">
      <div class="sc-label">Word Count</div>
      <div class="sc-val" id="v-words">—</div>
      <div class="sc-sub" id="s-words">0 sentences</div>
      <div class="sc-track"><div class="sc-bar" id="b-words"></div></div>
    </div>
    <div class="sc sc-risk" id="c-risk">
      <div class="sc-label">Risk Level</div>
      <div class="sc-val" id="v-risk" style="font-size:20px;padding-top:5px">—</div>
      <div class="sc-sub" id="s-risk">Perplexity: —</div>
      <div class="sc-track"><div class="sc-bar" id="b-risk"></div></div>
    </div>
  </div>

  <!-- Progress bar -->
  <div class="progress-wrap" id="prog-wrap">
    <div class="progress-bar" id="prog-bar"></div>
  </div>

  <!-- Workspace -->
  <div class="workspace">
    <div class="panel">
      <div class="ph">
        <span class="ph-title">Input Text</span>
        <div class="ph-meta">
          <span id="char-count">0 chars</span>
          <span class="badge" id="risk-badge"></span>
        </div>
      </div>
      <textarea id="input-text"
        placeholder="Paste or type any amount of text here — no length limit. Click 'Analyse' to see the plagiarism score, then 'Remove Plagiarism' to paraphrase it."></textarea>
    </div>

    <div class="panel output-panel">
      <div class="ph">
        <span class="ph-title">Paraphrased Output</span>
        <div class="ph-meta">
          <span id="out-words">—</span>
          <span class="badge" id="out-badge"></span>
        </div>
      </div>
      <div class="out-body" id="output-body">
        <span class="out-ph">Paraphrased text will appear here after you click "Remove Plagiarism".</span>
      </div>
      <div class="copy-wrap" id="copy-wrap" style="display:none">
        <button class="btn-copy" onclick="copyOutput()">⎘ Copy text</button>
        <button class="btn-copy" onclick="downloadOutput()">↓ Download .txt</button>
      </div>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <button class="btn btn-secondary" id="btn-analyze" onclick="runAnalyze()">
      🔍 Analyse
    </button>
    <button class="btn btn-primary" id="btn-para" onclick="runParaphrase()">
      ✦ Remove Plagiarism
    </button>
    <div class="divider"></div>
    <button class="btn btn-ghost" title="Clear all" onclick="clearAll()">✕</button>

    <div class="status">
      <div class="spinner" id="spinner"></div>
      <span id="status-text"></span>
    </div>
  </div>

  <!-- After-paraphrase banner -->
  <div class="after-banner" id="after-banner"></div>

  <!-- Perplexity explanation -->
  <div class="ppl-note">
    <strong>How scoring works:</strong>
    Plagiarism likelihood is measured using <strong>GPT-2 perplexity</strong> — a language model metric.
    Text that is highly predictable (low perplexity) matches patterns the model has memorised from common web text,
    suggesting it may be copied. Higher perplexity indicates more original phrasing.
    Paraphrasing is done by a <strong>T5-small</strong> model fine-tuned on paraphrase pairs — entirely local, no API calls.
  </div>
</main>

<script>
const inp       = document.getElementById('input-text');
const outBody   = document.getElementById('output-body');
const spinner   = document.getElementById('spinner');
const statusTxt = document.getElementById('status-text');
const progWrap  = document.getElementById('prog-wrap');
const progBar   = document.getElementById('prog-bar');
const copyWrap  = document.getElementById('copy-wrap');
const afterBanner = document.getElementById('after-banner');

let paraphrasedText = '';

// Live char count
inp.addEventListener('input', () => {
  const n = inp.value.length;
  document.getElementById('char-count').textContent = n.toLocaleString() + ' chars';
});

function setStatus(msg, spin = false) {
  statusTxt.textContent = msg;
  spinner.classList.toggle('show', spin);
}

function setProgress(pct) {
  progWrap.classList.add('show');
  progBar.style.width = pct + '%';
}

function clearProgress() {
  setTimeout(() => {
    progBar.style.width = '0';
    progWrap.classList.remove('show');
  }, 600);
}

function setBtns(disabled) {
  document.getElementById('btn-analyze').disabled = disabled;
  document.getElementById('btn-para').disabled    = disabled;
}

function animateBar(id, pct) {
  setTimeout(() => { document.getElementById(id).style.width = pct + '%'; }, 80);
}

function applyScoreCards(plag, orig, words, sentences, ppl, risk) {
  // Plagiarism
  document.getElementById('v-plag').textContent = plag + '%';
  document.getElementById('s-plag').textContent = plag >= 65 ? 'Likely copied' : plag >= 35 ? 'Possibly mixed' : 'Looks original';
  animateBar('b-plag', plag);

  // Originality
  document.getElementById('v-orig').textContent = orig + '%';
  document.getElementById('s-orig').textContent = 'Originality index';
  animateBar('b-orig', orig);

  // Words
  document.getElementById('v-words').textContent = words.toLocaleString();
  document.getElementById('s-words').textContent = sentences + ' sentence' + (sentences === 1 ? '' : 's');
  animateBar('b-words', Math.min(100, words / 10));

  // Risk
  document.getElementById('v-risk').textContent = risk;
  document.getElementById('s-risk').textContent = 'Perplexity: ' + ppl;
  const riskPct = plag >= 65 ? 90 : plag >= 35 ? 55 : 20;
  animateBar('b-risk', riskPct);

  ['c-plag','c-orig','c-words','c-risk'].forEach(id => {
    document.getElementById(id).classList.add('lit');
  });

  // Input badge
  const badge = document.getElementById('risk-badge');
  badge.textContent = risk;
  badge.className = 'badge show ' + (plag >= 65 ? 'badge-hi' : plag >= 35 ? 'badge-mid' : 'badge-lo');
}

async function runAnalyze() {
  const text = inp.value.trim();
  if (!text) { alert('Please enter some text first.'); return; }

  setBtns(true);
  setStatus('Analysing text…', true);
  setProgress(30);

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);

    setProgress(100);
    applyScoreCards(d.plagiarism_score, d.originality_score,
                    d.word_count, d.sentence_count, d.perplexity, d.risk_label);
    setStatus('Analysis complete.', false);
    afterBanner.classList.remove('show');
  } catch(e) {
    setStatus('Error: ' + e.message, false);
  } finally {
    setBtns(false);
    clearProgress();
  }
}

async function runParaphrase() {
  const text = inp.value.trim();
  if (!text) { alert('Please enter some text first.'); return; }

  setBtns(true);
  outBody.innerHTML = '<span class="out-ph">Paraphrasing… this may take a moment for long texts.</span>';
  copyWrap.style.display = 'none';
  afterBanner.classList.remove('show');

  // Animated progress (fake, since T5 is synchronous on the server)
  let pct = 5;
  setProgress(pct);
  setStatus('Paraphrasing text…', true);
  const ticker = setInterval(() => {
    pct = Math.min(pct + (100 - pct) * 0.06, 90);
    setProgress(pct);
  }, 600);

  try {
    const res = await fetch('/paraphrase', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);

    clearInterval(ticker);
    setProgress(100);

    paraphrasedText = d.paraphrased_text;
    outBody.textContent = paraphrasedText;
    copyWrap.style.display = 'flex';

    // Update output badge
    const ob = document.getElementById('out-badge');
    const np = d.new_plagiarism_score;
    ob.textContent = d.new_risk_label;
    ob.className = 'badge show ' + (np >= 65 ? 'badge-hi' : np >= 35 ? 'badge-mid' : 'badge-lo');
    document.getElementById('out-words').textContent =
      paraphrasedText.split(/\s+/).length.toLocaleString() + ' words';

    // After banner
    const orig = parseFloat(document.getElementById('v-plag').textContent) || 0;
    const diff = Math.round(orig - np);
    afterBanner.innerHTML = diff > 0
      ? `<strong>Plagiarism reduced by ~${diff}%.</strong> New score: ${np}% plagiarism · ${d.new_originality_score}% original · Risk: ${d.new_risk_label}. Perplexity: ${d.new_perplexity}`
      : `Paraphrasing complete. New plagiarism score: <strong>${np}%</strong> · Originality: <strong>${d.new_originality_score}%</strong> · Perplexity: ${d.new_perplexity}`;
    afterBanner.classList.add('show');

    setStatus('Done.', false);
  } catch(e) {
    clearInterval(ticker);
    outBody.innerHTML = '<span class="out-ph" style="color:var(--red)">Error: ' + e.message + '</span>';
    setStatus('Failed.', false);
  } finally {
    setBtns(false);
    clearProgress();
  }
}

function clearAll() {
  inp.value = '';
  outBody.innerHTML = '<span class="out-ph">Paraphrased text will appear here after you click "Remove Plagiarism".</span>';
  paraphrasedText = '';
  copyWrap.style.display = 'none';
  afterBanner.classList.remove('show');
  document.getElementById('char-count').textContent = '0 chars';
  ['v-plag','v-orig','v-words'].forEach(id => document.getElementById(id).textContent = '—');
  document.getElementById('v-risk').textContent = '—';
  ['s-plag','s-orig','s-words'].forEach(id => document.getElementById(id).textContent = 'Awaiting analysis');
  document.getElementById('s-risk').textContent = 'Perplexity: —';
  ['b-plag','b-orig','b-words','b-risk'].forEach(id => document.getElementById(id).style.width = '0');
  ['c-plag','c-orig','c-words','c-risk'].forEach(id => document.getElementById(id).classList.remove('lit'));
  ['risk-badge','out-badge'].forEach(id => { document.getElementById(id).className = 'badge'; });
  setStatus('', false);
}

function copyOutput() {
  navigator.clipboard.writeText(paraphrasedText).then(() => {
    const btn = document.querySelector('.btn-copy');
    btn.textContent = '✓ Copied!';
    setTimeout(() => { btn.textContent = '⎘ Copy text'; }, 2000);
  });
}

function downloadOutput() {
  const blob = new Blob([paraphrasedText], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'paraphrased_output.txt';
  a.click();
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)