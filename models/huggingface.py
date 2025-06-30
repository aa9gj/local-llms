from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set Hugging Face API Key from environment variable 
HUGGING_FACE_API_KEY = 'INPUT-YOUR-API-KEY'

# Model ID for ModernBERT-base
huggingface_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# List of required files fro the model 
required_files = [
    ".gitattributes",
    "config.json",
    "README.md",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "eval_results.json",
    "generation_config.json",
    "tokenizer.model"
]

# Download model files 
for filename in required_files:
    download_location = hf_hub_download(
        repo_id = huggingface_model,
        filename=filename,
        token=HUGGING_FACE_API_KEY
    )
    print(f"File downloaded to: {download_location}")

# Load the tokenizer and model 
model = AutoModelForCausalLM.from_pretrained(huggingface_model) #Enable trust_remote_code for safetensor
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)

# Create a pipline for text generation 
text_generation_pipeline = pipeline(
    "text-generation",
    model = model,
    tokenizer=tokenizer,
    max_length=1000
)

response = text_generation_pipeline("")
print(response)
