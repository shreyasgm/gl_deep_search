

def parse_unstructured(pdf_path: str, output_path: str = "unstructured_output.txt") -> dict:
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf(filename=pdf_path)
    text = "\n\n".join(str(el) for el in elements)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"engine": "unstructured", "text": text}




def parse_llamaparse(pdf_path: str, output_path: str = "llamaparse_output.md") -> dict:
    from llama_parse import LlamaParse
    parser = LlamaParse(result_type="markdown")  # or "text"
    with open(pdf_path, "rb") as f:
        documents = parser.load_data(f, extra_info={"file_name": pdf_path})
    text = "\n\n".join(doc.text for doc in documents)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"engine": "llamaparse", "text": text}




def parse_mistral(pdf_path: str, output_path: str = "mistral_output.md") -> dict:
    import base64
    import os
    from mistralai import Mistral
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not found in environment")
    with open(pdf_path, "rb") as f:
        encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded_pdf}"
        }
    )
    text = ocr_response.markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"engine": "mistral", "text": text}

