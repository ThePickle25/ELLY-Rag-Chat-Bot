import os
import json
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from uuid import uuid4
from crawl_data_pdf import crawl_data_from_pdf


def load_data(filename: str, directory: str) -> tuple:
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f'Data is loaded from {file_path}')
    return data, filename


def connect_to_milvus(URL_link: str, collection_name: str) -> Milvus:
    embeddings = OllamaEmbeddings(model='llama2')
    vertorstore = Milvus(
        embedding_function=embeddings,
        connection_args={'uri': URL_link},
        collection_name=collection_name,
    )
    return vertorstore


def seed_from_json(URL_link: str, collection_name: str, filename: str, directory: str) -> Milvus:
    embedding = OllamaEmbeddings(model='llama2')
    data, doc_name = load_data(filename, directory)
    document = [
        Document(
            page_content=doc.get("page_content") or '',
            metadata={
                'page': doc['metadata'].get('page')
            }
        )
        for doc in data
    ]
    print('Document: ', document)
    uuids = [str(uuid4()) for _ in range(len(document))]
    vectorstore = Milvus(
        embedding_function=embedding,
        connection_args={'uri': URL_link},
        collection_name=collection_name,
    )
    vectorstore.add_documents(documents=document, ids=uuids)
    print("vector: ", vectorstore)
    return vectorstore


def seed_from_pdf(PDF_path: str, URL_link: str, collection_name: str) -> Milvus:
    embedding = OllamaEmbeddings(model='llama2')
    document = crawl_data_from_pdf(PDF_path)
    for doc in document:
        metadata = {
            'page': doc.metadata.get('page') or " "
        }
        doc.metadata = metadata
    print('Document: ', document)
    uuids = [str(uuid4()) for _ in range(len(document))]
    vectorstore = Milvus(
        embedding_function=embedding,
        connection_args={'uri': URL_link},
        collection_name=collection_name,
        drop_old=True
    )
    vectorstore.add_documents(documents=document, ids=uuids)
    print("vector: ", vectorstore)
    return vectorstore


def main():
    """
    Hàm chính để kiểm thử các chức năng của module
    Thực hiện:
        1. Test seed_milvus với dữ liệu từ file local 'stack.json'
        2. (Đã comment) Test seed_milvus_live với dữ liệu từ trang web stack-ai
    Chú ý:
        - Đảm bảo Milvus server đang chạy tại localhost:19530
        - Các biến môi trường cần thiết (như OPENAI_API_KEY) đã được cấu hình
    """
    # Test seed_milvus với dữ liệu local
    seed_from_json('http://localhost:19530', 'data_test', 'doc.json', 'data')
    # Test seed_milvus_live với URL trực tiếp
    # seed_from_pdf('Thesis-AIP490_G9.docx-1.pdf', 'http://localhost:19530', 'data_test_live')


# Chạy main() nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main()
