from fastapi import UploadFile
from utils import save_upload_file_temp, remove_temp_file, transcribe_and_rag,rag_with_history
import os

async def process_text_question(question: str, pdf_file: UploadFile, chat_history=None):
    temp_pdf_path = f"temp_{pdf_file.filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(await pdf_file.read())
    
    # Process with RAG
    answer, updated_chat_history = rag_with_history(temp_pdf_path, question, chat_history)
    print('answer',answer)
    
    return {
        "answer": answer,
        "chat_history": updated_chat_history
    }

async def process_audio_question(pdf_file: UploadFile, audio_file: UploadFile):
    pdf_path = await save_upload_file_temp(pdf_file)
    audio_path = await save_upload_file_temp(audio_file)
    result = transcribe_and_rag(pdf_path, audio_path)
    remove_temp_file(pdf_path)
    remove_temp_file(audio_path)
    return {"answer": result}