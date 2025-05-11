# parse.py

import argparse
import json
import pathlib

'''
This script is used to parse a PDF using a selected engine.

The engines are:
- marker with llm
- unstructured
- llamaparse
'''

# also here I realised that my excel seems to have lost the columns for language lol. but not neededfor now

# Mock functions to replace with actual tool wrappers
def parse_marker(pdf_path):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    # Match existing structure exactly
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    text, _, images = text_from_rendered(rendered)

    return {"engine": "marker", "text": text}


def parse_unstructured(pdf_path):
    return {"engine": "unstructured", "text": "...unstructured output here..."}


def parse_llamaparse(pdf_path):
    return {"engine": "llamaparse", "text": "...llamaparse output here..."}


def parse_mistral(pdf_path):
    return {"engine": "mistral", "text": "...mistral output here..."}


PARSERS = {
    "marker": parse_marker,
    "unstructured": parse_unstructured,
    "llamaparse": parse_llamaparse,
    "mistral": parse_mistral,
}


def main():
    parser = argparse.ArgumentParser(description="Parse PDF using selected engine.")
    parser.add_argument(
        "--engine", choices=PARSERS.keys(), required=True, help="Parsing engine"
    )
    parser.add_argument(
        "--pdf", type=pathlib.Path, required=True, help="Path to input PDF"
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default="output.json", help="Output file path"
    )

    args = parser.parse_args()
    engine_func = PARSERS[args.engine]

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    result = engine_func(args.pdf)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Parsed using {args.engine}, saved to {args.out}")


if __name__ == "__main__":
    main()
