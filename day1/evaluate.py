from tenacity import retry, wait_chain, wait_fixed

import backends.openai
from testcases import run_test
import testcases.gsm8k

@retry(wait=wait_chain(*[wait_fixed(2*i+3) for i in range(10)]))
def prompt_openai_messages_with_retries(*args, **kwargs):
    return backends.openai.prompt_messages(*args, **kwargs)

r = run_test(
    testcases.gsm8k.get_test_dataset,
    testcases.gsm8k.preprocess_data,
    testcases.gsm8k.do_test,
    prompt_openai_messages_with_retries
)
