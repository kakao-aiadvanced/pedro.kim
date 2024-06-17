from tqdm import tqdm

def run_test(*args, **kwargs):
    return [result for result in run_test_streamed(*args, **kwargs)]

def run_test_streamed(
    dataset_loader,
    data_preprocessor,
    test_backend,
    prompt_backend
):
    try:
        dataset = dataset_loader()
        preprocessed_data = data_preprocessor(dataset)
        for tc in tqdm(preprocessed_data, total=len(preprocessed_data)):
            result = test_backend(tc, prompt_backend)
            yield result
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, stopping test")
