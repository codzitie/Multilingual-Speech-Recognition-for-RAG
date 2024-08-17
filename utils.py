import os
from fastapi import UploadFile
from config import client, embeddings, llm
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

async def save_upload_file_temp(upload_file: UploadFile) -> str:
    temp_file = f"temp_{upload_file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await upload_file.read())
    return temp_file

def remove_temp_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

def transcribe(audio_path):
    with open(audio_path, "rb") as file:
        translation = client.audio.translations.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            temperature=0.0
        )
    return translation.text

def transcribe_and_rag(content_pdf_path, question_path):
    question = transcribe(question_path)
    print("User Query Transcript", question)
    
    loader = PDFMinerLoader(content_pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)
    result = qa.run({"query": question})
    return result