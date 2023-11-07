import requests
from transformers import AutoTokenizer

API_TOKEN = "bsRCpGUjChYwaOikFsfWzOHqnGZkRiEUCwiipQrtjKuFkICSCRumZFPbGGqMLUJCIbHGpDIZxAsTjorlmUkVgbwBlUaUrbXrGnaucCFMZASwbRfeHzfZHIEdLpYSTTKb"

MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
LLAMA_MODEL = "meta-llama/Llama-2-70b-chat-hf"

MODEL = MISTRAL_MODEL

# MISTRAL_API_URL = f"https://api-inference.huggingface.co/models/{MISTRAL_MODEL}"
# LLAMA_API_URL = f"https://api-inference.huggingface.co/models/{LLAMA_MODEL}"
API_URL = "https://c1662ic0eolol8br.us-east-1.aws.endpoints.huggingface.cloud"

headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


tokenizer = AutoTokenizer.from_pretrained(MODEL, token=API_TOKEN)

response_type = 'detail_description'

system_message = open(f"prompts/{response_type}/system_message.txt", "r").read()

if MODEL is LLAMA_MODEL:
    messages = [{"role":"system","content":system_message}]
else:
    messages = [{"role":"user","content":system_message}, {"role":"assistant", "content":"Okay."}]

num_samples = 2
for s in range(num_samples):
    sample_caps = open(f"prompts/{response_type}/{s:03d}_caps.txt", "r").read()
    sample_conv = open(f"prompts/{response_type}/{s:03d}_conv.txt", "r").read()
    messages.append({"role":"user", "content":sample_caps})
    messages.append({"role":"assistant", "content":sample_conv})

messages.append({"role":"user","content":"A group of people standing outside of a black vehicle with various luggage.\nLuggage surrounds a vehicle in an underground parking area.\nPeople try to fit all of their luggage in an SUV.\nThe sport utility vehicle is parked in the public garage, being packed for a trip.\nSome people with luggage near a van that is transporting it\nperson: [0.681, 0.242, 0.774, 0.694]\n person: [0.63, 0.222, 0.686, 0.516]\n person: [0.444, 0.233, 0.487, 0.34]\n backpack: [0.384, 0.696, 0.485, 0.914]\n backpack: [0.755, 0.413, 0.846, 0.692]\n suitcase: [0.758, 0.413, 0.845, 0.69]\n suitcase: [0.1, 0.497, 0.173, 0.579]\n bicycle: [0.282, 0.363, 0.327, 0.442]\n car: [0.786, 0.25, 0.848, 0.322]\n car: [0.783, 0.27, 0.827, 0.335]\n car: [0.86, 0.254, 0.891, 0.3]\n car: [0.261, 0.101, 0.787, 0.626]"})
# messages.append({"role":"user","content":"A group of people standing outside of a black vehicle with various luggage.\nLuggage surrounds a vehicle in an underground parking area.\nPeople try to fit all of their luggage in an SUV.\nThe sport utility vehicle is parked in the public garage, being packed for a trip.\nSome people with luggage near a van that is transporting it"})

query_message = tokenizer.apply_chat_template(messages, tokenize=False)

# print(query_message)
print(repr(query_message))

output = query({
    "inputs": query_message,
    "parameters":{"max_new_tokens":1000}
})

print(output)

# for i in range(20):
#     output = query({
#         "inputs": output[0]['generated_text'],
#         "parameters": {"max_new_tokens": 100}
#     })
#     print(output)