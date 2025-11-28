import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken

BASE_PATH = "/mloscratch/homes/navasard/moe_doge/"
SPJ_DATA_PATH = os.path.join(BASE_PATH, "datasets/combined_slimpajama_debug_shuffled_full/")
FINEWEB_DATA_PATH = os.path.join(BASE_PATH, "datasets/fineweb_edu_bin/")

tknzr = tiktoken.get_encoding("gpt2")


def get_slimpajama_data(seq_len, batch_size, acc_steps, iterations, eval_freq, run_id=None, 
                        top_k=1, seed=1004, tgt_dataset=None, log_assignments=False):
    datasets_directory = "/mloscratch/homes/navasard/moe_doge/slimpajama"
    np.random.seed(seed)
    
    # === SlimPajama preprocessing (as before) ===
    if not os.path.exists(os.path.join(SPJ_DATA_PATH, "train.bin")):
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        datasets = [f for f in os.listdir(datasets_directory) 
                   if not os.path.isfile(os.path.join(datasets_directory, f))]
        splits = ['train', 'val']

        for split in splits:
            out_file = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            mem_data_list, mem_len_list = [], []        
            
            for ds in datasets:
                data = np.memmap(os.path.join(datasets_directory, ds, f"{split}.bin"), dtype=np.uint16, mode="r")
                mem_data_list.append(data)
                mem_len_list.append(len(data))

            sequences, source_ids = [], []
            for i, data in enumerate(tqdm(mem_data_list, desc=f"Loading {split} sequences")):
                num_sequences = len(data) // seq_len
                for seq_idx in range(num_sequences):
                    start, end = seq_idx * seq_len, (seq_idx + 1) * seq_len
                    sequences.append(data[start:end])
                    source_ids.append(i)
            
            shuffle_indices = np.arange(len(sequences))
            np.random.shuffle(shuffle_indices)
            
            total_tokens = len(sequences) * seq_len
            out_mem = np.memmap(out_file, dtype=np.uint16, mode="w+", shape=(total_tokens,))
            out_source_ids_mem = np.memmap(os.path.join(SPJ_DATA_PATH, f"{split}_source_ids.bin"), 
                                           dtype=np.uint16, mode="w+", shape=(total_tokens,))
            
            for i in tqdm(range(len(sequences)), desc=f"Writing shuffled {split}"):
                s_idx = shuffle_indices[i]
                start, end = i * seq_len, (i + 1) * seq_len
                out_mem[start:end] = sequences[s_idx]
                out_source_ids_mem[start:end] = np.full(seq_len, source_ids[s_idx], dtype=np.uint16)

            out_mem.flush()
            out_source_ids_mem.flush()

    # === Load SlimPajama binary data ===
    train_data = np.memmap(os.path.join(SPJ_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(SPJ_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r")

    train_source_ids = (
        np.memmap(os.path.join(SPJ_DATA_PATH, "train_source_ids.bin"), dtype=np.uint16, mode="r")
        if os.path.exists(os.path.join(SPJ_DATA_PATH, "train_source_ids.bin")) else None
    )
    val_source_ids = (
        np.memmap(os.path.join(SPJ_DATA_PATH, "val_source_ids.bin"), dtype=np.uint16, mode="r")
        if os.path.exists(os.path.join(SPJ_DATA_PATH, "val_source_ids.bin")) else None
    )

    # === Optional FineWeb Edu tgt_dataset loading ===
    tgt_train_data, tgt_val_data = None, None
    if tgt_dataset is not None and tgt_dataset.lower() == "fineweb-edu":
        if not os.path.exists(os.path.join(FINEWEB_DATA_PATH, "train.bin")):
            os.makedirs(FINEWEB_DATA_PATH, exist_ok=True)
            print("Preparing FineWeb‑Edu dataset...")

            # Load the dataset from Hugging Face
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")  # Adjust split/config as needed

            # FineWeb‑Edu has no predefined train/val, so we shuffle and split manually
            all_docs = dataset["train"]  # Full dataset
            # Shuffle docs for random split
            all_docs = all_docs.shuffle(seed=seed)

            split_ratio = 0.8  # 99% train, 1% validation
            split_idx = int(len(all_docs) * split_ratio)
            train_split = all_docs.select(range(split_idx))
            val_split = all_docs.select(range(split_idx, len(all_docs)))

            def process(example):
                # Tokenize text
                ids = tknzr.encode_ordinary(example["text"])
                ids.append(tknzr.eot_token)
                return {"ids": ids, "len": len(ids)}

            splits = {"train": train_split, "val": val_split}
            for split, dset in splits.items():
                # Map tokenization
                tokenized = dset.map(
                    process, 
                    remove_columns=dset.column_names, 
                    desc=f"Tokenizing FineWeb‑Edu {split}",
                    num_proc=8  # adjust number of processes
                )

                # Compute total length
                arr_len = np.sum(tokenized["len"])
                filename = os.path.join(FINEWEB_DATA_PATH, f"{split}.bin")
                arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
                idx = 0

                # Write tokens into memmap
                for batch_idx in tqdm(range(len(tokenized)), desc=f"Writing {split}"):
                    ids = np.array(tokenized[batch_idx]["ids"], dtype=np.uint16)
                    arr[idx:idx+len(ids)] = ids
                    idx += len(ids)
                arr.flush()

            print("FineWeb‑Edu dataset processed and saved")
        tgt_train_data = np.memmap(os.path.join(FINEWEB_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r")
        tgt_val_data = np.memmap(os.path.join(FINEWEB_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r")

    assert top_k >= 1, "top_k must be at least 1"

    train_len = batch_size * acc_steps * seq_len * iterations
    train_shape = (train_len, top_k) if top_k > 1 else (train_len,)

    assert iterations % eval_freq == 0, "Iterations must be divisible by eval_freq"
    max_num_batches = 24
    val_len = int(batch_size * max_num_batches * seq_len * (iterations / eval_freq - 1))
    val_shape = (val_len * 2, top_k) if top_k > 1 else (val_len * 2,)

    if log_assignments:
        os.makedirs(os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}"), exist_ok=True)
        train_exprt_dir = os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "train.bin")
        val_exprt_dir = os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "val.bin")
        train_exprt_mem = np.memmap(train_exprt_dir, dtype=np.int8, mode="w+", shape=train_shape)
        train_exprt_mem[:] = -1
        val_exprt_mem = np.memmap(val_exprt_dir, dtype=np.int8, mode="w+", shape=val_shape)
        val_exprt_mem[:] = -1
        train_exprt_mem.flush()
        val_exprt_mem.flush()

        train_loss = np.memmap(os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "train_loss.bin"), 
                            dtype=np.float32, mode="w+", shape=train_shape)
        val_loss = np.memmap(os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "val_loss.bin"), 
                            dtype=np.float32, mode="w+", shape=val_shape)
        train_loss[:] = -1.0
        val_loss[:] = -1.0
        train_loss.flush()
        val_loss.flush()

        return {
            "train": train_data[:train_len+1],
            "val": val_data[:val_len+1],
            "tgt_train": tgt_train_data[:train_len+1] if tgt_dataset else train_data[:train_len+1],
            "tgt_val": tgt_val_data[:val_len+1] if tgt_dataset else val_data[:val_len+1],
            "train_source_ids": train_source_ids,
            "val_source_ids": val_source_ids,
            "train_exp": train_exprt_mem if log_assignments else None,
            "val_exp": val_exprt_mem if log_assignments else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }


    return {
        "train": train_data[:train_len+1],
        "val": val_data[:val_len+1],
        "tgt_train": tgt_train_data[:train_len+1] if tgt_dataset else train_data[:train_len+1],
        "tgt_val": tgt_val_data[:val_len+1] if tgt_dataset else val_data[:val_len+1],
        "train_source_ids": train_source_ids,
        "val_source_ids": val_source_ids,
    }