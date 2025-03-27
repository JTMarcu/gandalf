
# Gandalf 🧙  
**LOTR Wisdom - Local Chatbot**

Gandalf is a fully local Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about "The Lord of the Rings" using LangChain, FAISS, and the Mistral-7B-Instruct model. This project ensures privacy by running entirely on your local machine, with no external API calls after setup.

## Features
- **Local Execution**: No internet or cloud APIs required after initial setup.
- **PDF Integration**: Processes "The Lord of the Rings" PDF for question answering.
- **Vector Search**: Uses FAISS for efficient document retrieval.
- **Mistral-7B-Instruct**: Runs the lightweight `mistral-7b-instruct-v0.1.Q4_K_M.gguf` model locally.
- **Gradio Interface**: User-friendly web interface for interacting with the chatbot.

## Project Structure
```
Gandalf/
├── gandalf_index/          # FAISS vectorstore files
│   ├── index.faiss
│   ├── index.pkl
├── models/mistral/         # Local Mistral model
│   ├── mistral-7b-instruct-v0.1.Q4_K_M.gguf
├── gandalf_mistral_local.ipynb  # Jupyter Notebook for local setup
├── gandalf_index.py        # Script to create FAISS index
├── app.py                  # Gradio app for chatbot
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup Instructions

### 1. Install Required Libraries
Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Environment
- Place your "The Lord of the Rings" PDF in the project directory.
- Update the `.env` file with your Hugging Face API token:
  ```properties
  HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
  ```

### 3. Create the FAISS Index
Run the `gandalf_index.py` script to process the PDF and create the FAISS vectorstore:
```bash
python gandalf_index.py
```

### 4. Run the Chatbot
You can use the chatbot in two ways:
1. **Gradio Interface**: Launch the web app using `app.py`:
   ```bash
   python app.py
   ```
   Open the provided URL in your browser to interact with Gandalf.
2. **Jupyter Notebook**: Open gandalf_mistral_local.ipynb or gandalf_chatbot_demo.ipynb and follow the steps.

## How It Works
1. **Document Processing**: The PDF is split into chunks using LangChain's text splitter.
2. **Vector Embedding**: Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index.
3. **Question Answering**: The Mistral-7B-Instruct model retrieves relevant chunks and generates answers.

## Example Usage
Ask Gandalf a question:
```python
question = "What happened in the mines of Moria?"
result = qa_chain.invoke({"query": question})
print("🧙 Gandalf says:\n", result['result'])
```

## Requirements
- Python 3.8+
- Sufficient disk space for the Mistral model and FAISS index.

## Notes
- The Mistral model file (`mistral-7b-instruct-v0.1.Q4_K_M.gguf`) is optimized for local inference.
- Ensure you have sufficient memory (RAM) to load the model.

## Future Enhancements
- Add support for additional texts from Tolkien's legendarium.
- Experiment with other lightweight local models for improved performance.
- Enhance the Gradio interface with additional features like context visualization.

---
**"A wizard is never late, nor is he early, he arrives precisely when he means to."**
```