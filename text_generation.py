from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import FinishReason
from transformers import AutoTokenizer
import requests
from enum import Enum

import parse_responses


class Models(Enum):
    MISTRAL = "mistralai/Mistral-7B-Instruct-v0.1"
    LLAMA = "meta-llama/Llama-2-70b-chat-hf"


API_TOKEN = "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"
ENDPOINT_API_TOKEN = "bsRCpGUjChYwaOikFsfWzOHqnGZkRiEUCwiipQrtjKuFkICSCRumZFPbGGqMLUJCIbHGpDIZxAsTjorlmUkVgbwBlUaUrbXrGnaucCFMZASwbRfeHzfZHIEdLpYSTTKb"
INFERENCE_ENDPOINT_URL = {Models.MISTRAL: "https://c1662ic0eolol8br.us-east-1.aws.endpoints.huggingface.cloud"}

models_with_system_messages = {Models.LLAMA}

APITypes = Enum("APITypes", ["INFERENCE_API", "INFERENCE_ENDPOINT"])


def get_api_url_and_token(model, api_type):
    if api_type == APITypes.INFERENCE_API:
        return model.value, API_TOKEN
    else:
        return INFERENCE_ENDPOINT_URL[model], ENDPOINT_API_TOKEN



# def get_api_url(model):
#     if model == Models.MISTRAL:
#         return "https://c1662ic0eolol8br.us-east-1.aws.endpoints.huggingface.cloud"
#     else:
#         return None

class ResponseTypes(Enum):
    COMPLEX_REASONING = 'complex_reasoning'
    CONVERSATION = 'conversation'
    DETAIL_DESCRIPTION = 'detail_description'
    COMPLEX_REASONING_PRUNING = 'complex_reasoning_pruning'


# get the corresponding number of samples for the given response type
response_n_shot = {ResponseTypes.COMPLEX_REASONING: 4, ResponseTypes.CONVERSATION: 2,
                   ResponseTypes.DETAIL_DESCRIPTION: 3}


class TextGenerator():
    def __init__(self, model, api_type, max_new_tokens=500):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.value, token=API_TOKEN)
        model_url, model_token = get_api_url_and_token(model, api_type)
        self.client = InferenceClient(model=model_url, token=model_token)

        self.max_new_tokens = max_new_tokens

        self.base_messages = {}
        for response_type in ResponseTypes:

            system_message = open(f"prompts/{response_type.value}/system_message.txt", "r").read()

            if self.model in models_with_system_messages:
                messages = [{"role": "system", "content": system_message}]
            else:
                messages = [{"role": "user", "content": system_message}, {"role": "assistant", "content": "Okay."}]

            num_samples = response_n_shot[response_type]

            for s in range(num_samples):
                sample_caps = open(f"prompts/{response_type.value}/{s:03d}_caps.txt", "r").read()
                sample_conv = open(f"prompts/{response_type.value}/{s:03d}_conv.txt", "r").read()
                messages.append({"role": "user", "content": sample_caps})
                messages.append({"role": "assistant", "content": sample_conv})

            self.base_messages[response_type] = messages

    def query(payload):
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        return response.json()

    def generate(self, query, response_type):

        query_message = self.tokenizer.apply_chat_template(self.base_messages[response_type] + [{"role": "user", "content": query}], tokenize=False)

        output = self.client.text_generation(query_message, max_new_tokens=self.max_new_tokens, details=True)

        return output.generated_text, output.details.finish_reason

    def generate_complex_reasoning(self, query):
        query_message = self.tokenizer.apply_chat_template(self.base_messages[ResponseTypes.COMPLEX_REASONING_PRUNING] + [{"role": "user", "content": query}], tokenize=False)
        potential_questions = []
        for i in range(5):
            output = self.client.text_generation(query_message, temperature=0.7, max_new_tokens=50, details=True)

            print(output.generated_text)

            qa = parse_responses.parse_conversation(output.generated_text)
            if qa:
                potential_questions.append(qa['Question'])

        for question in potential_questions: