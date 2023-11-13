from transformers import AutoModelForCausalLM,AutoTokenizer
import json
from enum import Enum
import re
from tqdm import tqdm


# using gpu (I believe this doesn't split across gpus so use below if you need to parallelize across multiple)
device = "cuda"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model.to(device)
# Alternatively split load across hardware instead using:
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map = 'auto')

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

class ResponseTypes(Enum):
    COMPLEX_REASONING = 'complex_reasoning'
    CONVERSATION = 'conversation'
    DETAIL_DESCRIPTION = 'detail_description'
    COMPLEX_REASONING_PRUNING = 'complex_reasoning_pruning'

def cut_off_eos(text):
    eos_index = text.find("</s>")
    # check to make sure that finish reason is end of string token
    if eos_index == -1:
        raise ValueError("Max tokens generated")
    return text[:eos_index]


def parse_conversation(conv_raw):
    question_or_answers = re.split(r'\n===\n', conv_raw.strip())
    qa_pairs = []
    for q, a in zip(question_or_answers[0::2], question_or_answers[1::2]):
        if not q.startswith('Question:\n') or not a.startswith('Answer:\n'):
            raise ValueError("Conversation ill-formed")
        qa_pairs.append({'Question': q[10:], 'Answer': a[8:]})

    if len(qa_pairs) == 0:
        raise ValueError("Conversation ill-formed")
    return qa_pairs


BATCH_SIZE = 2
response_types_to_generate = [ResponseTypes.COMPLEX_REASONING,ResponseTypes.CONVERSATION,ResponseTypes.DETAIL_DESCRIPTION]
max_new_tokens = 750

for response_type in response_types_to_generate:
    with open(f"prompts/{response_type.value}.json","r") as f:
        queries = json.load(f)

    print(f'loaded {response_type.value} queries')

    all_parsed_responses = []
    for i in tqdm(range(0, len(queries), BATCH_SIZE)):
        # gather batch from queries
        batch = queries[i:i + BATCH_SIZE]
        batch_text = [b['prompt'] for b in batch]

        # the batch is already in chat template, just need to turn it into a tensor with token ids and obtain attention mask
        tokenized_batch = tokenizer(batch_text, return_tensors="pt", padding=True, add_special_tokens=False)
        # length of the longest sequence of token ids in the batch
        batch_token_length = tokenized_batch['input_ids'].shape[1]
        model_inputs = tokenized_batch.to(device)

        # batched inference
        generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens)
        # get only the new token ids that were generated
        new_ids = generated_ids[:,batch_token_length:]
        response_texts = tokenizer.batch_decode(new_ids)

        for i,response_text in enumerate(response_texts):
            # ensure generated responses are correct
            try:
                # cut off eos tokens
                response_text_no_eos = cut_off_eos(response_text)

                # parse response according to its type (conversation style or not)
                parsed_response = response_text_no_eos if (response_type == ResponseTypes.CONVERSATION or response_type == ResponseTypes.COMPLEX_REASONING) \
                    else parse_conversation(response_text_no_eos)

                response_object = {'image_id':batch[i]['image_id'], 'response':parsed_response}
                all_parsed_responses.append(response_object)

            # update a json newline file so that no generated data is lost on crash or program fail
                with open(f"log/{response_type.value}.json", "a") as jf:
                    json.dump(response_object, jf, indent=2)
                    jf.write(',\n')

            except ValueError as e:
                # print error message to an error log json
                with open(f"error_log.json", "a") as ef:
                    json.dump({'image_id': batch[i]['image_id'], 'text': response_text, 'error': str(e)}, ef, indent=2)
                    ef.write(',\n')

    with open(f"dataset/{response_type.value}.json", "w") as jf:
        json.dump(all_parsed_responses, jf)