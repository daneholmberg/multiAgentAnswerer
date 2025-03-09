#!/usr/bin/env python3
import sys
import argparse
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent AI Collaboration Tool")
    parser.add_argument("question", type=str, help="The question to process")
    args = parser.parse_args()

    sys.exit(main(args.question))
