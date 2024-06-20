from datasets import load_dataset

# download huggingface datasets and prepare it for megatron
num_procs =48
dataset_name = 'openwebtext'

if __name__ == '__main__':
    ds = load_dataset(dataset_name, split="train", keep_in_memory=False)
    ds.to_json(f"dataset/{dataset_name}.jsonl", orient="records", lines=True, force_ascii=False)
