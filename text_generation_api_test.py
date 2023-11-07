from huggingface_hub import InferenceClient
import requests
from transformers import AutoTokenizer

API_TOKEN = "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"

MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LLAMA_MODEL = "meta-llama/Llama-2-70b-chat-hf"

MODEL = LLAMA_MODEL

MISTRAL_API_URL = f"https://api-inference.huggingface.co/models/{MISTRAL_MODEL}"
LLAMA_API_URL = f"https://api-inference.huggingface.co/models/{LLAMA_MODEL}"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

tokenizer = AutoTokenizer.from_pretrained(MODEL, token=API_TOKEN)

def query(payload):
    response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
    return response.json()

system_message = open("prompts/conversation/system_message.txt", "r").read()

if MODEL is LLAMA_MODEL:
    messages = [{"role":"system","content":system_message}]
else:
    messages = [{"role":"user","content":system_message}, {"role":"assistant", "content":"Okay."}]

num_samples = 2
for s in range(num_samples):
    sample_caps = open(f"prompts/conversation/{s:03d}_caps.txt", "r").read()
    sample_conv = open(f"prompts/conversation/{s:03d}_conv.txt", "r").read()
    messages.append({"role":"user", "content":sample_caps})
    messages.append({"role":"assistant", "content":sample_conv})

messages.append({"role":"user","content":"A group of people standing outside of a black vehicle with various luggage.\nLuggage surrounds a vehicle in an underground parking area.\nPeople try to fit all of their luggage in an SUV.\nThe sport utility vehicle is parked in the public garage, being packed for a trip.\nSome people with luggage near a van that is transporting it"})

query_message = tokenizer.apply_chat_template(messages, tokenize=False)

# print(query_message)
# print(repr(query_message))

# output = query({
#     "inputs": query_message,
#     "parameters":{"max_new_tokens":500}
# })
#
# print(output)


client = InferenceClient(model=MODEL, token=API_TOKEN)

output = client.text_generation(query_message,max_new_tokens=10,details=True)

from huggingface_hub.inference._text_generation import FinishReason
print(output)
print(output.details)
# print(output.details.finish_reason)
# print(output.details.finish_reason == FinishReason.Length)
