from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from seed_data import seed_from_pdf
from agent import get_agent, get_retriever
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods='*',
    allow_headers='*'
)

MILVUS_URL = os.getenv("MILVUS_URL")
TEMP_UPLOAD_DIR = "./uploaded_files"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


@app.post("/seed_upload_pdf")
async def seed_upload_pdf(file: UploadFile = File(...), collection_name: str = Form(...)):
    try:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        seed_from_pdf(file_path, MILVUS_URL, collection_name)
        return {'message': 'Seed PDF to Milvus successfully!'}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post("/seed_url_pdf")
async def seed_url_pdf(file: str = Form(...), collection_name: str = Form(...)):
    try:
        seed_from_pdf(file, MILVUS_URL, collection_name)
        return {'message': 'Seed PDF to Milvus successfully!'}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post("/chat")
async def chat_with_doc(prompt: str = Form(...), collection_name: str = Form(...), history: Optional[str] = Form(default=[])):
    try:
        retriever = get_retriever(collection_name, MILVUS_URL)
        agent_executor = get_agent(retriever)
        # chat_history = [{'role': msg['role'], 'content': msg['content']} for msg in history]
        chat_history = json.loads(history)
        response = agent_executor.invoke({
            'input': prompt,
            'chat_history': chat_history
        })
        return {'answer': response['output']}
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
