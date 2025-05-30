from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
import os

# Load variables from .env into environment
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI-API")
print('grok',GROQ_API_KEY)
MODEL_PATH = "sentence-transformers/all-MiniLM-l6-v2"
client = Groq(api_key=GROQ_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",  
#     temperature=0.1,
#     convert_system_message_to_human=True 
# )
# llm = ChatOllama(model="llama3.2")
# llm = ChatOllama(model="deepseek-ai_-_deepseek-coder-33b-base-8bits")