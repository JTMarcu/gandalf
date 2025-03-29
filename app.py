import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Note: In a Hugging Face Space, the token should be set as an environment variable in the Space settings.
from dotenv import load_dotenv # remove when on HF Space

# 🔐 Load Hugging Face API token
load_dotenv() #remove line when on HF Space. 
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ Missing Hugging Face API token in environment variables.")

# 📚 Load FAISS vectorstore from uploaded folder (must be uploaded to Space)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("gandalf_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# 🤖 Load LLM from Hugging Face Inference API
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=hf_token)

llm = HuggingFaceEndpoint(
    client=client,
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7,
    max_new_tokens=256,
)

# 📜 Create custom prompt and QA chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question. 
If the answer is not in the context, reply as Gandalf would — something wise or witty, and make it clear that the answer is unknown.
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

# 🎛️ Gradio interface
import gradio as gr

def ask_gandalf(question):
    # Retrieve relevant docs
    docs = retriever.get_relevant_documents(question)

    # Get unique chapters from docs
    chapters = {doc.metadata.get("chapter", "Unknown") for doc in docs}
    chapter_info = ", ".join(sorted(chapters))

    # Build custom prompt with context + chapter info
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = custom_prompt.format(
        context=context,
        question=question,
        chapter_info=chapter_info or "Unknown"
    )

    # Call the model
    answer = llm.invoke(prompt)

    return f"{answer}\n\n📖 Referenced Chapter(s): {chapter_info}"

demo = gr.Interface(
    fn=ask_gandalf,
    inputs=gr.Textbox(placeholder="Ask Gandalf anything about Middle-earth..."),
    outputs="text",
    title="Ask Gandalf",
    description="🧙 Ask questions based on the lore from The Lord of the Rings."
)

if __name__ == "__main__":
    demo.launch()