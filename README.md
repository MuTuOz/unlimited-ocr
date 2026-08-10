# ocr-rag

Containerised document pipeline: **Unlimited-OCR → FAISS → chat model**.

Built to be reproducible. Every dependency is pinned, the model weights live
on a mounted volume, and the GPU phase is separated from the CPU phase so a
failure in one does not cost you the other.

---

## Prerequisites

On the **host** (not in the container):

```bash
nvidia-smi                      # must print your GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If the second command fails, install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Everything else is inside the image.

At least **16 GB VRAM** is comfortable; the OCR model alone is ~7.5 GB in
fp16. It runs on a 14.5 GB T4 as long as nothing else is on the card.

## Setup

```bash
cp .env.example .env       # then put your Groq key in it
docker build -t ocr-rag .
```

First build pulls the CUDA base image, so expect a few minutes. Rebuilds after
a code change take seconds — `requirements.txt` is copied before `src/`, so the
pip layer stays cached.

## Usage

**Check the environment first.** This catches the majority of problems before
you spend hours on OCR:

```bash
docker run --rm --gpus all -v ./models:/models --env-file .env ocr-rag doctor
```

**Phase 1 — OCR (GPU, slow, run once):**

```bash
docker run --rm --gpus all \
  -v ./models:/models -v ./data:/data \
  ocr-rag ocr --input /data/mydoc.pdf --output /data/ocr_out
```

Resumes automatically. Kill it, restart it, it skips pages already on disk.
For an 889-page PDF expect several hours; run it under `tmux` or with
`-d` and follow with `docker logs -f`.

**Phase 2 — index (CPU, fast):**

```bash
docker run --rm -v ./models:/models -v ./data:/data \
  ocr-rag index --ocr-dir /data/ocr_out --index /data/index
```

**Phase 3 — ask:**

```bash
docker run --rm -it -v ./models:/models -v ./data:/data --env-file .env \
  ocr-rag ask --index /data/index --question "what is the test pressure?"
```

Omit `--question` for an interactive prompt. Answers cite the page numbers
they came from.

**No LLM needed:** for a single page or a short document, skip retrieval
entirely — the OCR output *is* the answer:

```bash
docker run --rm -v ./data:/data ocr-rag dump --ocr-dir /data/ocr_out
```

## Tuning OCR quality

If tables come out half-empty or words are garbled (`A bevground Piping`
instead of `Aboveground Piping`), the input resolution is too low:

```bash
--dpi 400 --base-size 1024 --image-size 1024
```

Two settings that actively hurt on forms and are off by default here:

- `no_repeat_ngram_size` — suppresses legitimate repetition. Forms repeat
  "PROJECT.", "Date", "Signature" by design; penalising that blanks out cells.
- `--image-size 640` — too coarse for dense tables.

**Before OCR-ing anything, check whether you need to.** If the PDF already has
a text layer, `PyMuPDF` extracts it in seconds and OCR is wasted effort:

```python
import fitz
print(repr(fitz.open("mydoc.pdf")[10].get_text()[:300]))
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'langchain.chains'` | langchain 1.x moved it | pinned to 0.3.7 here |
| `cannot import name 'is_torch_fx_available'` | transformers 5.x | pinned to 4.46.3 here |
| `cannot import name 'find_pruneable_heads_and_indices'` | transformers 5.x via sentence-transformers | fastembed used instead |
| `Found no NVIDIA driver` | `--gpus all` missing | add it, verify with `doctor` |
| `CUDA out of memory` | two model copies loaded | one process per phase; `doctor` shows pre-load VRAM |
| `RateLimitError` / `RESOURCE_EXHAUSTED` | no API credit | use `dump`, no key required |
| `Illegal header value` | newline inside the API key | `.env` must be one line, no trailing paste |

## Layout

```
Dockerfile           CUDA 12.4 + torch 2.5.1 base, pinned deps
requirements.txt     every version pinned, with a comment on why
docker-compose.yml   ocr (GPU) and rag (CPU) services
src/ocr.py           model loading, PDF rendering, resume, VRAM hygiene
src/rag.py           chunking, FAISS, retrieval chain
src/main.py          CLI
data/                your documents in, results out  (gitignored)
models/              HF cache, survives rebuilds     (gitignored)
```

## Notes

- `models/` is a volume so the 7.5 GB download happens once, not per build.
- Embeddings use `fastembed` (ONNX), not `sentence-transformers`. The latter
  requires transformers 5.x, which breaks the OCR model. This is the single
  most important constraint in the whole dependency set.
- FAISS is the CPU build on purpose — the index is small, and the GPU build
  would compete with OCR for VRAM.
- To swap the chat model, set `CHAT_MODEL` in `.env`. Any OpenAI-compatible
  endpoint works by editing `_chat_model()` in `src/rag.py` — useful if your
  organisation runs an internal vLLM or Azure OpenAI deployment.
