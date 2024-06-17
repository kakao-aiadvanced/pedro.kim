from tqdm import tqdm

def run_test(*args, **kwargs):
    return [result for result in run_test_streamed(*args, **kwargs)]

def run_test_streamed(
    dataset_loader,
    data_preprocessor,
    test_backend,
    prompt_backend,
    stop_after=None
):
    try:
        dataset = dataset_loader()
        processed_data = data_preprocessor(dataset)
        num_cases = len(processed_data)
        if isinstance(stop_after, int) and stop_after < num_cases:
            num_cases = stop_after
        for n, tc in enumerate(tqdm(processed_data, total=num_cases)):
            result = test_backend(tc, prompt_backend)
            yield result
            if n+1 >= num_cases:
                break
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, stopping test")
