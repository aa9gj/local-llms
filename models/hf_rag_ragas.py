import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset

# --- LangChain & RAG Components ---
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Ragas Evaluation Components ---
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

# --- 0. SCRIPT CONFIGURATION ---
# The document you want to use for the RAG system.
KNOWLEDGE_BASE_FILE = "final_sentences.txt"

# --- CHECK FOR KNOWLEDGE FILE ---
if not os.path.exists(KNOWLEDGE_BASE_FILE):
    print(f"Error: The file '{KNOWLEDGE_BASE_FILE}' was not found.")
    print("Please make sure the file exists in the same directory as the script.")
    exit()

# --- 1. SETUP: LOAD YOUR LOCAL LLM AND EMBEDDING MODEL ---

print("--- Setting up models (this may take a while on the first run) ---")

# A. Load the local LLM (TinyLlama)
# This is the model that will generate answers.
huggingface_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading LLM: {huggingface_model}...")

tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
model = AutoModelForCausalLM.from_pretrained(
    huggingface_model,
    trust_remote_code=True,
    torch_dtype=torch.float32  # Explicitly use float32 for CPU
)

# Create a transformers pipeline for text generation.
# This pipeline will be used for both the RAG chain and the Ragas evaluation.
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id,
    truncation=True,  # Ensure prompts are truncated to fit the model's context window
    max_length=2048   # Max context window for TinyLlama
)

# Wrap the pipeline in a LangChain LLM object.
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("LLM setup complete.")

# B. Load the embedding model (runs locally)
# This model converts text into numerical vectors for similarity searching.
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading embedding model: {embedding_model_name}...")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
print("Embedding model setup complete.")


# --- 2. INDEXING: LOAD DATA AND CREATE A VECTOR STORE ---

def create_vector_db(file_path: str, embeddings_model):
    """Loads a text file, splits it into chunks, and creates a FAISS vector store."""
    print(f"\n--- Creating vector database from {file_path} ---")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    print("Vector store created successfully.")
    return vector_store

# Create the vector store from your text file
vector_store = create_vector_db(KNOWLEDGE_BASE_FILE, embeddings)


# --- 3. RETRIEVAL & GENERATION: CREATE THE RAG CHAIN ---

print("\n--- Setting up RAG chain ---")
retriever = vector_store.as_retriever()
RAG_PROMPT_TEMPLATE = """
Answer the question based only on the provided context. If the context does not contain the answer, say that you don't know.

<context>
{context}
</context>

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("RAG chain setup complete.")


# --- 4. GENERATE DATA FOR EVALUATION ---

print("\n--- Generating data for Ragas evaluation ---")

# !!! IMPORTANT !!!
# You MUST replace these with questions and answers relevant to YOUR document.
eval_questions = [
    "What is the main topic of the document?"
]
eval_ground_truths = [
    "The main topic is foods and compounds and their associations with disease."
]

eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

for i, question in enumerate(eval_questions):
    print(f"Generating answer for: '{question}'")
    response = retrieval_chain.invoke({"input": question})
    
    # Store the generated answer and retrieved context 
    answer = response.get("answer", "")
    contexts_list = [doc.page_content for doc in response.get("context", [])]

    eval_data["question"].append(question)
    eval_data["answer"].append(response.get("answer", ""))
    eval_data["contexts"].append(contexts_list)
    eval_data["ground_truth"].append(eval_ground_truths[i])

# Convert the dictionary to a Hugging Face Dataset
dataset = Dataset.from_dict(eval_data)
print("Evaluation dataset created:")
print(dataset)


# --- 5. RUN RAGAS EVALUATION ---

print("\n--- Running RAGAS Evaluation (this will take several minutes) ---")

# Configure the timeout for Ragas evaluation jobs.
# This is crucial for preventing timeouts when running on a CPU.
custom_run_config = RunConfig(timeout=600)  # 300 seconds = 5 minutes

# Define the metrics for evaluation
metrics = [
    faithfulness,
    context_precision,
    context_recall,
]

# Run the evaluation
try:
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,                     # The same LLM is used as the critic
        embeddings=embeddings,       # The same embeddings are used
        run_config=custom_run_config
    )
    print("\n--- RAGAS Evaluation Results ---")
    print(score)

except Exception as e:
    print(f"\nAn error occurred during RAGAS evaluation: {e}")

