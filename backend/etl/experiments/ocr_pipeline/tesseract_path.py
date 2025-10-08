import shutil
import subprocess

import pytesseract

# Explicitly set tesseract binary location (adjust the path if your username is different)
pytesseract.pytesseract.tesseract_cmd = "/n/home04/kdaryanani/local/bin/tesseract"

# Print the path pytesseract will use
print("Tesseract binary:", pytesseract.pytesseract.tesseract_cmd)
print("which tesseract:", shutil.which("tesseract"))

# Print Tesseract version as seen by pytesseract
try:
    version_output = subprocess.check_output(
        [pytesseract.pytesseract.tesseract_cmd, "--version"], text=True
    )
    print("Tesseract version output:\n", version_output)
except Exception as e:
    print("Failed to run tesseract:", e)
