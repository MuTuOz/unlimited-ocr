"""OCR phase: document -> one markdown file per page, written to disk.

Design notes (these exist because of specific failures, don't undo them):

* Resume by default. An 889-page PDF takes hours; if the process dies at
  page 600 you restart and it skips the 599 pages already on disk.
* Pages are rendered to PNG in small batches and deleted after OCR.
  Rendering all 889 at 300 DPI up front is ~5-10 GB of scratch space.
* torch.cuda.empty_cache() after every page. Without it, allocator
  fragmentation builds up and you OOM somewhere in the middle of a long run.
* The model is loaded exactly once, by whoever constructs OcrEngine. Never
  construct it twice in one process -- that is what put two 7.5 GB copies on
  a 14.5 GB card.
"""

from __future__ import annotations

import gc
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("OCR_MODEL", "baidu/Unlimited-OCR")

# The OCR model emits layout tags like <|det|>header [x, y, x, y]<|/det|>.
# Useful to keep in the raw dump, noise for embeddings.
DET_TAG = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)

# save_results=True has written .md, .txt and .mmd depending on version.
TEXT_SUFFIXES = (".md", ".mmd", ".txt")


@dataclass
class OcrSettings:
    """Defaults mirror the configuration that works on this PDF.

    crop_mode is OFF. With crop_mode=True the model tiles the source image,
    and a 300 DPI page (~2480x3508) produces enough tiles that activations
    push past 15 GB of VRAM on a T4. image_size does NOT control the tile
    count -- turning crop_mode off is what bounds the memory.

    Pass --crop only for single images that are
    already small, where tiling buys detail on dense tables.
    """

    dpi: int = 300
    base_size: int = 1024
    image_size: int = 1024
    crop_mode: bool = False
    max_length: int = 8192
    prompt: str = "<image>document parsing."
    render_batch: int = 25


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class OcrEngine:
    """Owns the OCR model. Construct once per process."""

    def __init__(self, settings: OcrSettings | None = None) -> None:
        self.settings = settings or OcrSettings()
        self.device = _device()

        if self.device == "cpu":
            log.warning(
                "No CUDA device visible. OCR will run on CPU and take minutes "
                "per page. Did you pass --gpus all to docker run?"
            )

        # float16 on purpose: T4 (Turing) has no hardware bfloat16, so bf16 is
        # emulated and slower. On A100/H100 you can switch this to bfloat16.
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        log.info("Loading %s onto %s (%s)", MODEL_NAME, self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=dtype,
            )
            .eval()
            .to(self.device)
        )
        log.info("Model ready (%.1f GB allocated)", self._vram_gb())

    # -- public ----------------------------------------------------------

    def process(self, source: Path, out_dir: Path, resume: bool = True) -> list[Path]:
        """OCR a PDF or a single image. Returns the per-page markdown paths."""
        out_dir.mkdir(parents=True, exist_ok=True)

        if source.suffix.lower() == ".pdf":
            return self._process_pdf(source, out_dir, resume)
        return self._process_image(source, out_dir, resume)

    # -- pdf -------------------------------------------------------------

    def _process_pdf(self, pdf: Path, out_dir: Path, resume: bool) -> list[Path]:
        doc = fitz.open(pdf)
        total = len(doc)
        log.info("%s: %d pages", pdf.name, total)

        written: list[Path] = []
        mat = fitz.Matrix(self.settings.dpi / 72, self.settings.dpi / 72)
        batch = self.settings.render_batch

        for start in range(0, total, batch):
            stop = min(start + batch, total)
            scratch = Path(tempfile.mkdtemp(prefix="pages_"))
            try:
                for idx in range(start, stop):
                    target = out_dir / f"page_{idx + 1:04d}.md"
                    if resume and target.exists() and target.stat().st_size > 0:
                        written.append(target)
                        continue

                    png = scratch / f"page_{idx + 1:04d}.png"
                    doc[idx].get_pixmap(matrix=mat).save(png)

                    text = self._infer_one(png)
                    target.write_text(text, encoding="utf-8")
                    written.append(target)

                    png.unlink(missing_ok=True)
                    log.info(
                        "page %d/%d  (%d chars, %.1f GB VRAM)",
                        idx + 1,
                        total,
                        len(text),
                        self._vram_gb(),
                    )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)

        doc.close()
        return written

    # -- single image ----------------------------------------------------

    def _process_image(self, image: Path, out_dir: Path, resume: bool) -> list[Path]:
        target = out_dir / f"{image.stem}.md"
        if resume and target.exists() and target.stat().st_size > 0:
            log.info("%s already done, skipping", target.name)
            return [target]

        text = self._infer_one(image)
        target.write_text(text, encoding="utf-8")
        log.info("%s -> %s (%d chars)", image.name, target.name, len(text))
        return [target]

    # -- internals -------------------------------------------------------

    def _infer_one(self, image_path: Path) -> str:
        scratch = Path(tempfile.mkdtemp(prefix="ocr_out_"))
        try:
            with torch.inference_mode():
                self.model.infer(
                    self.tokenizer,
                    prompt=self.settings.prompt,
                    image_file=str(image_path),
                    output_path=str(scratch),
                    base_size=self.settings.base_size,
                    image_size=self.settings.image_size,
                    crop_mode=self.settings.crop_mode,
                    max_length=self.settings.max_length,
                    save_results=True,
                )
            return self._collect(scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            # Do this every page, not every run. Fragmentation is cumulative.
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @staticmethod
    def _collect(scratch: Path) -> str:
        """Read whatever save_results wrote.

        Older code paths write .md, some builds write .mmd. If nothing matches
        we log the directory contents instead of silently returning "" -- a
        silent empty string is how you end up indexing 889 blank pages.
        """
        parts: list[str] = []
        for path in sorted(scratch.iterdir()):
            if path.suffix.lower() in TEXT_SUFFIXES:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))

        if not parts:
            listing = [p.name for p in scratch.iterdir()]
            log.warning("No text output found. Files present: %s", listing)
            return ""

        return "\n\n".join(parts).strip()

    def _vram_gb(self) -> float:
        if self.device != "cuda":
            return 0.0
        return torch.cuda.memory_allocated() / 1e9

    def unload(self) -> None:
        """Free the 7.5 GB so a second model can be loaded in-process."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def strip_layout_tags(text: str) -> str:
    """Remove <|det|>...<|/det|> markers, keep the content and tables."""
    return DET_TAG.sub("", text).strip()
