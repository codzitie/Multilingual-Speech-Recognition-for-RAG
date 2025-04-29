import os
from fastapi import UploadFile
from config import client, embeddings, llm
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


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

def rag_with_history(content_pdf_path, question, chat_history=None):
    """
    RAG function that maintains chat history between interactions
    
    Args:
        content_pdf_path: Path to the PDF file
        question: Current user question
        chat_history: List of tuples of (human_message, ai_message) from previous interactions
    
    Returns:
        answer: The response
        new_chat_history: Updated chat history
    """
    print('Processing PDF:', content_pdf_path)
    
    # Initialize chat history if None
    if chat_history is None:
        chat_history = []

    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, including any important context from the conversation history."""
    
    # Load and process the PDF
    loader = PDFMinerLoader(content_pdf_path)
    data = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    # Create vector store
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    
    # Create memory object to store chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Add existing chat history to memory
    for human_msg, ai_msg in chat_history:
        memory.chat_memory.add_user_message(human_msg)
        memory.chat_memory.add_ai_message(ai_msg)
    
    # Create conversational chain with memory
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        # condense_question_prompt= template,
    )
    
    # Run the chain
    result = qa({"question": question})
    answer = result["answer"]
    
    # Update chat history
    chat_history.append((question, answer))
    
    return answer, chat_history