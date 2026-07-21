"""Launch the GUI: python -m gui [--share] [--port 7860]"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Georgian Dub Studio")
    p.add_argument("--share", action="store_true",
                   help="create a public gradio.live link (default in Colab)")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    from .app import build_app

    build_app().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
