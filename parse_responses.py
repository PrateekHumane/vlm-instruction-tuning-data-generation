import re
def parse_conversation(conv_raw):
    question_or_answers = re.split(r'\n===\n', conv_raw.strip())
    qa_pairs = []
    for q, a in zip(question_or_answers[0::2], question_or_answers[1::2]):
        if not q.startswith('Question:\n') or not a.startswith('Answer:\n'):
            return None
        qa_pairs.append({'Question': q[10:], 'Answer': a[8:]})

    return qa_pairs

# print(parse_conversation(open(f"prompts/conversation/000_conv.txt", "r").read()))