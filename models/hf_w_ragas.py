import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset

# RAGAS related imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall
)

# LangChain components for Hugging Face models
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Model ID for TinyLlama-1.1B-Chat-v1.0
huggingface_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_model_dir = f"./{huggingface_model.replace('/', '_')}"

# Download model files (this part is fine, but you only need to run it once)
# You can comment this out after the first successful download.
print("Downloading model files if not already cached...")
# The from_pretrained method below will handle downloading and caching automatically.
# The manual download loop is not strictly necessary unless you need to control the location precisely.

print("Loading tokenizer and model...")
# Using local_dir ensures it loads from your downloaded files.
# If the files are not found, it will download them to the default cache directory.
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
model = AutoModelForCausalLM.from_pretrained(huggingface_model, trust_remote_code=True)


text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,      # Max tokens for the *generated output*
    max_length=2048,         # Max tokens for the *entire sequence (prompt + output)*
    truncation=True,         # <-- ADD THIS to truncate long prompts
    pad_token_id=tokenizer.eos_token_id
)

print("\n--- Performing Text Generation ---")
question = "What is cancer?"
# Using a chat template is better for chat models
chat_template = [
    {"role": "user", "content": question},
]
prompt = tokenizer.apply_chat_template(chat_template, tokenize=False)
raw_response = text_generation_pipeline(prompt)
generated_answer = raw_response[0]['generated_text']

# Clean the generated answer
final_answer = generated_answer[len(prompt):].strip()


print(f"Question: {question}")
print(f"Generated Answer: {final_answer}")


# --- START OF RAGAS EVALUATION BLOCK ---

print("\n--- Preparing Data for RAGAS Evaluation ---")

contexts_for_cancer = [
    "Cancer is a disease in which some of the body’s cells grow uncontrollably and spread to other parts of the body.",
    "As cells become more and more abnormal, old or damaged cells survive when they should die, and new cells form when they are not needed.",
    "These extra cells can divide without stopping and may form growths called tumors."
]

data = {
    'question': [question],
    'answer': [final_answer],
    'contexts': [contexts_for_cancer],
    'ground_truth': ["Cancer is a disease characterized by the uncontrolled growth and spread of abnormal cells."],
}

dataset = Dataset.from_dict(data)

print("Dataset for RAGAS:")
print(dataset)

print("\n--- Running RAGAS Evaluation ---")

# --- Configure LLM and Embeddings for RAGAS (Critic Models) ---
# 1. Setup the LLM for RAGAS (critic LLM)
# Use LangChain's HuggingFacePipeline to wrap your transformers pipeline.
# NO extra RagasLLM wrapper is needed.
critic_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 2. Setup Embeddings for RAGAS.
# NO extra RagasEmbedding wrapper is needed.
ragas_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Define the metrics you want to use.
# Note: ResponseRelevancy is not included as it doesn't use context.
metrics = [
    faithfulness,
    context_precision,
    context_recall,
]

# Evaluate the dataset, passing the LangChain objects directly.
try:
    score = evaluate(
        dataset,
        metrics=metrics,
        llm=critic_llm,             # Pass the LangChain LLM instance directly
        embeddings=ragas_embeddings # Pass the LangChain Embeddings instance directly
    )
    print("\n--- RAGAS Evaluation Results ---")
    print(score)

except Exception as e:
    print(f"\nAn error occurred during RAGAS evaluation: {e}")
