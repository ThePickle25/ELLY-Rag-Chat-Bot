import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz
import io
import torch
from PIL import Image
import os
from typing import List
import google.generativeai as genai


# ---Config---
OUTPUT_DIR = ""
CHUNK_SIZE = 1000
OVERLAP = 200
GEMINI_API = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")

device = "cuda" if torch.cuda.is_available() else 'cpu'


def image_describer(image: Image.Image, input_text: str = "", model_name=None, api_key=None):
    if api_key is None:
        api_key = GEMINI_API

    if model_name is None:
        model_name = MODEL_NAME

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return "Failed to caption picture!"
    try:
        prompt = input_text
        response = model.generate_content([prompt, image], stream=True)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"Failed to caption picture: {e}")
        return "Failed to caption picture"


def extract_page_element(page):
    element = []
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if "lines" in b:
            text = "".join(span["text"] for line in b["lines"] for span in line["spans"])
            element.append({"type": "text", "bbox": fitz.Rect(b["bbox"]), "content": text})

    for img_info in page.get_images(full=True):
        xref, x0, y0, x1, y1 = img_info[0], img_info[1], img_info[2], img_info[3], img_info[4]
        pix = fitz.Pixmap(page.parent, xref)
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        element.append({"type": "image", "bbox": fitz.Rect(x0, y0, x1, y1), "content": img})

    element.sort(key=lambda e: e["bbox"].y0)
    return element


def crawl_data_from_pdf(pdf_path: str) -> List[Document]:
    data = fitz.open(pdf_path)
    document = []
    for page_num, page in enumerate(data, start=1):
        element = extract_page_element(page)
        full_text = ""
        for idx, elem in enumerate(element):
            if elem["type"] == "text":
                full_text += elem["content"] + "\n"
            elif elem["type"] == "image":
                pre_text = post_text = ""
                for prev in reversed(element[:idx]):
                    if prev['type'] == 'text':
                        pre_text = prev['content'][-30:] + " " + pre_text
                        break
                for post in element[idx:]:
                    if post["type"] == "text":
                        post_text = post['content'][:30]
                        break
                context = f"Describe the image based on the nearby text: {pre_text.strip()} ... {post_text.strip()}"
                full_text += image_describer(elem['content'], context)

        document.append(Document(page_content=full_text.strip(), metadata={"page": page_num}))

    document_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP
    )
    split_document = document_splitter.split_documents(document)
    return split_document


def save_data_to_local(documents, output_dir, file_name):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_path = os.path.join(output_dir, file_name)
    data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Data is saved to {file_path}')


def save_to_txt(documents, output_dir, file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)
    data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(file_path, 'w') as f:
        f.write(data['page_content'])
    print(f'Data is saved to {file_path}')


def main():
    pdf_path = "Thesis-AIP490_G9.docx-1.pdf"
    document = crawl_data_from_pdf(pdf_path)
    save_data_to_local(document, 'data', 'doc.json')
    # save_to_txt(document, 'data', 'doc.txt')
    print('data: ', document)


if __name__ == "__main__":
    main()
