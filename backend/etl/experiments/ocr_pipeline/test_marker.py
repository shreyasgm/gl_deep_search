# test_marker.py

from module_marker import parse_marker_preset

# Replace this with the actual path to a PDF you want to test (relative to your current dir)
pdf_path = "data/test_data/2024-03-cid-wp-442-japan-economic-puzzle.pdf"

result = parse_marker_preset(
    pdf_path,
    preset="baseline",            # Try also: 'use_llm', etc.
    model="none",                 # Or "gpt-4o-mini" if using LLMs and you have an API key
    output_path="/n/hausmann_lab/lab/kdaryanani/deeplearn/gl_deep_search/backend/etl/experiments/ocr_pipeline/marker.md"              # Or give a path for output
)

print(result)