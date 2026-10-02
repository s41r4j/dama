#!/usr/bin/env python3
from dama.cli import main
raise SystemExit(main(["train", *__import__("sys").argv[1:]]))
