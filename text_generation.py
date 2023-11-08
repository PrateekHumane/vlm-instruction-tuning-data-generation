import math

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


class ResponseTypes(Enum):
    COMPLEX_REASONING = 'complex_reasoning'
    CONVERSATION = 'conversation'
    DETAIL_DESCRIPTION = 'detail_description'
    COMPLEX_REASONING_PRUNING = 'complex_reasoning_pruning'


# get the corresponding number of samples for the given response type
response_n_shot = {ResponseTypes.COMPLEX_REASONING: 4, ResponseTypes.CONVERSATION: 2,
                   ResponseTypes.DETAIL_DESCRIPTION: 3, ResponseTypes.COMPLEX_REASONING_PRUNING: 5}


class TextGenerator():
    def __init__(self, model, api_type, max_new_tokens=500):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.value, token=API_TOKEN)
        if api_type == APITypes.INFERENCE_API:
            self.model_url, self.model_token = model.value, API_TOKEN
        else:
            self.model_url, self.model_token = INFERENCE_ENDPOINT_URL[model], ENDPOINT_API_TOKEN

        self.client = InferenceClient(model=self.model_url, token=self.model_token)

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

    def http_query_api(self, payload):
        api_url = f"https://api-inference.huggingface.co/models/{self.model_url}"
        response = requests.post(api_url, headers={"Authorization": f"Bearer {self.model_token}"}, json=payload)
        return response.json()

    def generate(self, query, response_type):

        query_message = self.tokenizer.apply_chat_template(
            self.base_messages[response_type] + [{"role": "user", "content": query}], tokenize=False)

        output = self.client.text_generation(query_message, max_new_tokens=self.max_new_tokens, details=True)

        return output.generated_text, output.details.finish_reason

    def generate_complex_reasoning_pruned(self, query):
        query_message = self.tokenizer.apply_chat_template(
            self.base_messages[ResponseTypes.COMPLEX_REASONING] + [{"role": "user", "content": query}],
            tokenize=False)

        potential_questions = []
        for i in range(5):
            output = self.http_query_api({
                "inputs": query_message,
                "parameters": {"return_full_text": False, "temperature": 0.8, "max_new_tokens": 50},
                "options": {"use_cache": False}
            })[0]['generated_text']

            print(output)

            qa = parse_responses.parse_conversation(output)
            if qa is not None or len(qa) > 0:
                potential_questions.append(qa[0]['Question'])

        # TODO: ensure there is at least 1 question generated and parsed
        query_questions = [f"{i}. {question}" for i, question in enumerate(potential_questions,1)]
        # query2 = query + '\n\n' + '\n'.join(query_questions)
        query2 = '\n'.join(query_questions)

        query2_message = self.tokenizer.apply_chat_template(
            self.base_messages[ResponseTypes.COMPLEX_REASONING_PRUNING] + [{"role": "user", "content": query2}],
            tokenize=False)
        output = self.client.text_generation(query2_message, max_new_tokens=self.max_new_tokens, details=True)

        # TODO: return proper errors
        if not output.generated_text.isnumeric():
            print(f'failed w output: {output}')
            return None

        best_question_num = int(output.generated_text) - 1
        # TODO: return proper errors
        if not 0 <= best_question_num < len(potential_questions):
            print(f'failed w range: {output}')
            return None

        print(output.details.tokens[0].logprob)
        print(math.exp(output.details.tokens[0].logprob))

        output = self.client.text_generation(query_message+' Question:\n'+potential_questions[best_question_num], max_new_tokens=self.max_new_tokens, details=True)

        return 'Question:\n'+potential_questions[best_question_num]+output.generated_text, output.details.finish_reason