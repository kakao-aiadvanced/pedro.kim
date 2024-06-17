import backends.openai
from testcases import run_test
import testcases.gsm8k

run_test(
    "gsm8k_testresult.txt",
    testcases.gsm8k.get_test_dataset,
    testcases.gsm8k.preprocess_data,
    testcases.gsm8k.do_test,
    backends.openai.prompt_messages
)
