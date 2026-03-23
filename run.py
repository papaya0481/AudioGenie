from __future__ import annotations
import argparse, json, os, pathlib
import os
import argparse
import json
import pathlib
from router import load_llm
from agents import AudioGenieSystem


os.environ['GEMINI_API_KEY'] = 'Your_Gemini_Api_Key'


def main():
    parser = argparse.ArgumentParser(description="AudioGenie (training-free multi-agent)")
    parser.add_argument("--text", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--video", default=None, help="Path to .mp4 video file.")
    parser.add_argument("--outdir", default="/hpc2hdd/home/yrong854/jhaidata/Agent/outputs_gemini/bird_sea")
    parser.add_argument("--llm", default="google_gemini")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_siblings", type=int, default=1)
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir or "outputs")
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    llm = load_llm(args.llm)

    ctx = {
        "text": args.text,
        "image": args.image,
        "video": args.video if args.video not in ("None", "none", "") else None,
    }

    system = AudioGenieSystem(llm, outdir=outdir)
    out = system.run(ctx, max_depth=args.max_depth, max_siblings=args.max_siblings)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()