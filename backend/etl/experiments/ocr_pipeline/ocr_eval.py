import difflib
import os
import re
from difflib import SequenceMatcher

import editdistance
import nltk
from jiwer import Compose, RemovePunctuation, ToLowerCase, cer, wer

nltk.download("punkt", quiet=True)

MD_DIR = "."  # Set to current directory
GT_FILE = "atlas_2013_part1_paste.md"
OCR_FILE = "atlas_2013_part1_unstructured.md"


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def remove_markdown(md):
    # Remove Markdown links, headers, emphasis, images etc. (very basic)
    text = re.sub(r"\[.*?\]\(.*?\)", "", md)  # Remove [text](link)
    text = re.sub(r"\!\[.*?\]\(.*?\)", "", text)  # Remove ![img](src)
    text = re.sub(r"\#+ ", "", text)  # Remove headers
    text = re.sub(r"[`*_>#+-]", "", text)  # Remove common markup
    text = re.sub(r"\n{2,}", "\n", text)  # Collapse blank lines
    return text


def print_terminal_diff(gt_lines, ocr_lines, n=50):
    print("\n=== CLI Unified Diff (first few lines) ===")
    diff = list(
        difflib.unified_diff(
            gt_lines, ocr_lines, fromfile=GT_FILE, tofile=OCR_FILE, lineterm=""
        )
    )
    print("\n".join(diff[:n]))
    if len(diff) > n:
        print(f"... ({len(diff) - n} more diff lines truncated)")


def write_html_diff(gt_lines, ocr_lines, out_path):
    html_diff = difflib.HtmlDiff().make_file(
        gt_lines, ocr_lines, fromdesc="GT", todesc="OCR"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_diff)


def compute_sentence_error_rate_aligned(gt, ocr):
    gt_sents = [s.strip() for s in nltk.sent_tokenize(gt) if s.strip()]
    ocr_sents = [s.strip() for s in nltk.sent_tokenize(ocr) if s.strip()]
    sm = SequenceMatcher(None, gt_sents, ocr_sents)
    n_total = len(gt_sents)
    n_errors = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # Any replacement, insert or delete is a sentence-level error.
        n_errors += max(i2 - i1, j2 - j1)
    return n_errors / n_total if n_total else 0


def main():
    gt_path = os.path.join(MD_DIR, GT_FILE)
    ocr_path = os.path.join(MD_DIR, OCR_FILE)
    gt_md = load_text(gt_path)
    ocr_md = load_text(ocr_path)

    # --- Optional: Strip Markdown
    gt = remove_markdown(gt_md)
    ocr = remove_markdown(ocr_md)

    # --- Normalization for fair comparison
    norm = Compose([ToLowerCase(), RemovePunctuation()])

    # --- Metrics
    print("\n=== Plaintext Comparison (raw) ===")
    print(f"Chars in GT:  {len(gt)} | OCR: {len(ocr)}")
    print(f"WER: {wer(gt, ocr):.4f}")
    print(f"CER: {cer(gt, ocr):.4f}")
    print(f"Levenshtein: {editdistance.eval(gt, ocr)}")
    print(f"Similarity: {difflib.SequenceMatcher(None, gt, ocr).ratio():.4f}")

    print("\n=== Normalized Comparison (lower, no punct) ===")
    gt_norm = norm(gt)
    ocr_norm = norm(ocr)
    print(f"WER (norm): {wer(gt_norm, ocr_norm):.4f}")
    print(f"CER (norm): {cer(gt_norm, ocr_norm):.4f}")
    print(f"Levenshtein (norm): {editdistance.eval(gt_norm, ocr_norm)}")
    print(
        f"Similarity (norm): {difflib.SequenceMatcher(None, gt_norm, ocr_norm).ratio():.4f}"
    )

    gt_lines = gt.splitlines()
    ocr_lines = ocr.splitlines()
    print_terminal_diff(gt_lines, ocr_lines)
    write_html_diff(gt_lines, ocr_lines, "diff_report.html")
    print("HTML side-by-side diff written to: diff_report.html")
    print("Open it in your browser to explore!")

    # In your main:
    ser = compute_sentence_error_rate_aligned(gt, ocr)
    print(f"Sentence Error Rate (aligned): {ser:.4f}")


if __name__ == "__main__":
    main()
