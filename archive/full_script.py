import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in your .env file.")

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = "Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(splits, embedding_model)
retriever = db.as_retriever()

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=hf_token)

llm = HuggingFaceEndpoint(
    client=client,
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7,
    max_new_tokens=256,
)

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question. 
If you don't know the answer, just say you don't know — do not make up an answer.

Context:
{context}

Question:
{question}

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

question = "What happened in the mines of Moria?"
result = qa_chain.invoke({"query": question})
print("🧙 Gandalf says:\n", result['result'])