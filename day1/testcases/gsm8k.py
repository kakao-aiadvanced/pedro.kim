import requests

from backends import SystemPrompt, UserPrompt
from datasets import load_dataset

_cot_hub_cache = {}

def get_test_dataset():
    gsm8k = load_dataset('gsm8k', 'main')
    gsm8k_test = gsm8k['test']
    return gsm8k_test

def preprocess_data(dataset):
    return dataset

def do_test_question_only(testcase, prompt_backend):
    q, a = testcase['question'], testcase['answer']
    messages = [
        SystemPrompt("Answer the following question."),
        UserPrompt(f"Q: {q}")
    ]
    result = prompt_backend(messages)
    return {
        "question": q,
        "answer": a,
        "answer_model": result
    }

def do_test_cot_complex_prompt(testcase, prompt_backend):
    prompt_base = _cot_hub_cache.get("complex_prompt")
    if prompt_base is None:
        prompt_base = requests.get(
            "https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/"
            "main/gsm8k/lib_prompt/prompt_hardest.txt"
        ).text
        _cot_hub_cache["complex_prompt"] = prompt_base
    q, a = testcase['question'], testcase['answer']
    messages = [
        SystemPrompt("Follow the given examples and answer the question."),
        UserPrompt(f"{prompt_base}\n\nQuestion: {q}\n")
    ]
    result = prompt_backend(messages)
    return {
        "question": q,
        "answer": a,
        "answer_model": result
    }
