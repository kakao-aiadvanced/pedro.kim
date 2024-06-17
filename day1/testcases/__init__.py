from tqdm import tqdm

def run_test(
    output_file_path,
    dataset_loader,
    data_preprocessor,
    test_backend,
    prompt_backend
):
    dataset = dataset_loader()
    preprocessed_data = data_preprocessor(dataset)
    with open(output_file_path, "w") as f:
        for tc in tqdm(preprocessed_data, total=len(preprocessed_data)):
            result = test_backend(tc, prompt_backend)
            f.write(result)
