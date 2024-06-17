import sys

from tenacity import retry, wait_chain, wait_fixed

import backends.openai
from testcases import run_test
import testcases.gsm8k

@retry(wait=wait_chain(*[wait_fixed(2*i+3) for i in range(10)]))
def prompt_openai_messages_with_retries(*args, **kwargs):
    return backends.openai.prompt_messages(*args, **kwargs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: {} <testname>".format(sys.argv[0]))
        sys.exit(1)

    test_suites = {
        "gsm8k": (
            testcases.gsm8k.get_test_dataset,
            testcases.gsm8k.preprocess_data,
            testcases.gsm8k.do_test_question_only,
        ),
        "gsm8k_cot_complex": (
            testcases.gsm8k.get_test_dataset,
            testcases.gsm8k.preprocess_data,
            testcases.gsm8k.do_test_cot_complex_prompt,
        )
    }
    test_suite = test_suites.get(sys.argv[1], None)

    if test_suite is None:
        print("valid test names are: {}".format(",".join(test_suites)))
        sys.exit(1)

    dataset_loader, data_preprocessor, test_backend = test_suite

    test_results = run_test(
        dataset_loader,
        data_preprocessor,
        test_backend,
        prompt_openai_messages_with_retries
    )
