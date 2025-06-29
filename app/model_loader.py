from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

repo_name = "pramod350/fine-tuned-mistral"
hf_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(repo_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    repo_name, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    token=hf_token
)
