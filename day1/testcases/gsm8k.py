from backends import SystemPrompt, UserPrompt
from datasets import load_dataset

def get_test_dataset():
    gsm8k = load_dataset('gsm8k', 'main')
    gsm8k_test = gsm8k['test']
    return gsm8k_test

def preprocess_data(dataset):
    return dataset

def do_test(testcase, prompt_backend):
    q, a = testcase['question'], testcase['answer']
    messages = [
        SystemPrompt("Follow the given examples and answer the question."),
        UserPrompt(f"Q: {q}")
    ]
    result = prompt_backend(messages)
    return f"A_model: {result}\nA: {a}\n--------\n"
