from fastapi import UploadFile
from utils import save_upload_file_temp, remove_temp_file, transcribe_and_rag

async def process_text_question(question: str, pdf_file: UploadFile):
    pdf_path = await save_upload_file_temp(pdf_file)
    result = transcribe_and_rag(pdf_path, question)
    remove_temp_file(pdf_path)
    return {"answer": result}

async def process_audio_question(pdf_file: UploadFile, audio_file: UploadFile):
    pdf_path = await save_upload_file_temp(pdf_file)
    audio_path = await save_upload_file_temp(audio_file)
    result = transcribe_and_rag(pdf_path, audio_path)
    remove_temp_file(pdf_path)
    remove_temp_file(audio_path)
    return {"answer": result}