import os
from fastapi import UploadFile
from config import client, embeddings, llm
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document

# Global variable to store chat history
global_chat_history = []
# Global variable to store the initialized retriever (to avoid reprocessing the PDF each time)
global_retriever = None

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

def create_parent_document_retriever(docs, embeddings, k=4):
    """Best for maintaining context while enabling precise retrieval"""
    # Create parent and child splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs1 = [Document(page_content=" Hello ")]
    # Vector store and storage
    vectorstore = FAISS.from_documents(docs1, embeddings)  # Empty initially
    store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        k=k
    )
    
    # Add documents
    retriever.add_documents(docs)
    return retriever


def rag_with_history(content_pdf_path, question,  reset_history=False):
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
    
    global global_chat_history
    global global_retriever
    
    # Reset history if requested
    if reset_history:
        global_chat_history = []
        global_retriever = None

    if global_retriever is None:
        # Load and process the PDF
        loader = PDFMinerLoader(content_pdf_path)
        data = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)
        
        # # Create vector store
        # db = FAISS.from_documents(docs, embeddings)
        # global_retriever = db.as_retriever()

        global_retriever = create_parent_document_retriever(docs, embeddings)

    condense_question_prompt = PromptTemplate.from_template("""
    Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question that captures all relevant context from the conversation.
    
    Chat History:
    {chat_history}
    
    Follow Up Question: {question}
    
    Standalone Question:
    """)
    
    qa_prompt = PromptTemplate.from_template("""
    You are a helpful assistant answering questions based on provided documents. Those documents are research papers.
    Use the following pieces of retrieved context to answer the question. If you don't know the 
    answer, just say you don't know. Use the previous conversation to provide continuity in your answers.

    Question: {question}
    
    Context: {context}
    
    Chat History: {chat_history}
    
    Answer:
    """)
    literature_review_prompt = PromptTemplate.from_template("""
    You are a research assistant tasked with writing a detailed **literature review** based on the content of retrieved documents.
    The documents are scientific research papers. Your goal is to:

    1. Identify prior work and studies mentioned in the paper.
    2. Analyze how the current paper compares with or builds on past work.
    3. Highlight research gaps that the paper aims to address.
    4. Summarize key contributions from the referenced literature.
    5. Maintain an academic tone and avoid speculation.

    Use the provided context to produce a well-structured literature review. If the context lacks sufficient references to past work, say so.

    ---  
    Context:  
    {context}  

    ---  
    Literature Review:
    """)
    # Create memory object to store chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Add existing chat history to memory
    for human_msg, ai_msg in global_chat_history:
        memory.chat_memory.add_user_message(human_msg)
        memory.chat_memory.add_ai_message(ai_msg)
    
    # Create conversational chain with memory
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=global_retriever,
        memory=memory,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    # Run the chain
    result = qa({"question": question})
    answer = result["answer"]
    
    # Update chat history
    global_chat_history.append((question, answer))
    
    return answer, global_chat_history