"""CLI. Phases are separate commands on purpose.

    ocr     GPU, slow, run once      -> markdown on disk
    index   CPU, fast                -> FAISS index on disk
    ask     CPU + API                -> answer
    dump    nothing but a file read  -> raw OCR text
    doctor  environment sanity check

Keeping them separate is what stops the "three layers all broken at once"
problem: if `ask` fails you do not re-run four hours of OCR.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These two are chatty and say nothing useful during a long OCR run.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)


def cmd_doctor(_: argparse.Namespace) -> int:
    import torch
    import transformers

    print(f"python        {sys.version.split()[0]}")
    print(f"torch         {torch.__version__}")
    print(f"transformers  {transformers.__version__}")

    ok = True

    if not transformers.__version__.startswith("4."):
        print("  !! transformers must be 4.x for the OCR remote code")
        ok = False

    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa: F401

        print("  is_torch_fx_available  present")
    except ImportError:
        print("  !! is_torch_fx_available missing -> OCR model will not load")
        ok = False

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"cuda          {name}  {total:.1f} GB total, {used:.1f} GB in use")
        if used > 0.5:
            print("  !! memory already allocated before loading anything")
            ok = False
    else:
        print("cuda          NOT AVAILABLE (did you pass --gpus all?)")
        ok = False

    import os

    print(f"GROQ_API_KEY  {'set' if os.environ.get('GROQ_API_KEY') else 'MISSING'}")
    print(f"HF_HOME       {os.environ.get('HF_HOME', '(default)')}")

    print("\nOK" if ok else "\nProblems found above.")
    return 0 if ok else 1


def cmd_ocr(args: argparse.Namespace) -> int:
    from .ocr import OcrEngine, OcrSettings

    source = Path(args.input)
    if not source.exists():
        print(f"Not found: {source}", file=sys.stderr)
        return 1

    engine = OcrEngine(
        OcrSettings(
            dpi=args.dpi,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=not args.no_crop,
            max_length=args.max_length,
        )
    )
    written = engine.process(source, Path(args.output), resume=not args.no_resume)
    print(f"{len(written)} pages written to {args.output}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from .rag import build_index

    n = build_index(Path(args.ocr_dir), Path(args.index), chunk_size=args.chunk_size)
    print(f"{n} chunks indexed into {args.index}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .rag import ask, make_chain

    chain = make_chain(Path(args.index), k=args.k)

    if args.question:
        answer, pages = ask(chain, args.question)
        print(answer)
        if pages:
            print(f"\n[pages: {', '.join(map(str, pages))}]")
        return 0

    print("Interactive mode. Blank line or Ctrl-D to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        answer, pages = ask(chain, q)
        print(f"\n{answer}")
        if pages:
            print(f"[pages: {', '.join(map(str, pages))}]")
        print()


def cmd_dump(args: argparse.Namespace) -> int:
    from .rag import dump

    text = dump(Path(args.ocr_dir))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"{len(text)} chars written to {args.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ocr-rag", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check versions, GPU and keys")
    d.set_defaults(func=cmd_doctor)

    o = sub.add_parser("ocr", help="PDF/image -> markdown per page (GPU)")
    o.add_argument("--input", required=True, help="/data/mydoc.pdf or .png")
    o.add_argument("--output", required=True, help="/data/ocr_out")
    o.add_argument("--dpi", type=int, default=300)
    o.add_argument("--base-size", type=int, default=1024)
    o.add_argument("--image-size", type=int, default=1024)
    o.add_argument("--max-length", type=int, default=8192)
    o.add_argument("--no-crop", action="store_true")
    o.add_argument("--no-resume", action="store_true", help="redo finished pages")
    o.set_defaults(func=cmd_ocr)

    i = sub.add_parser("index", help="markdown -> FAISS index (CPU)")
    i.add_argument("--ocr-dir", required=True)
    i.add_argument("--index", required=True)
    i.add_argument("--chunk-size", type=int, default=1200)
    i.set_defaults(func=cmd_index)

    a = sub.add_parser("ask", help="query the index")
    a.add_argument("--index", required=True)
    a.add_argument("--question", help="omit for interactive mode")
    a.add_argument("-k", type=int, default=4)
    a.set_defaults(func=cmd_ask)

    u = sub.add_parser("dump", help="print OCR text, no LLM, no API key")
    u.add_argument("--ocr-dir", required=True)
    u.add_argument("--output")
    u.set_defaults(func=cmd_dump)

    return p


def main() -> int:
    args = build_parser().parse_args()
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
