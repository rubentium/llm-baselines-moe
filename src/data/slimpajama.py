from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os 


SPJ_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/combined_slimpajama_debug/")
SPJ_CHUNK_1_DATA_PATH = os.path.join(SPJ_DATA_PATH, "chunk1")


tknzr = tiktoken.get_encoding("gpt2")


def get_slimpajama_data(num_proc=40):
    if not os.path.exists(os.path.join(SPJ_DATA_PATH, "train.bin")):
        # Concatenate all train val files into one large file and also create source_ids
        # which contains the index of the dataset in the original SlimPajama dataset
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        datasets_directory = "/mloscratch/homes/navasard/moe_doge/slimpajama_dbg"
        datasets = [f for f in os.listdir(datasets_directory) if not os.path.isfile(os.path.join(datasets_directory, f))]
        splits = ['train', 'val']

        for split in splits:
            out_file = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            mem_data_list = []
            mem_len_list = []        
            for i in range(len(datasets)):
                data = np.memmap(os.path.join(datasets_directory, datasets[i], f"{split}.bin"), dtype=np.uint16, mode="r")
                mem_data_list.append(data)
                mem_len_list.append(len(data))
            
            out_mem = np.memmap(out_file, dtype=np.uint16, mode="w+", shape=(sum(mem_len_list)))
            if split == 'train':
                out_source_ids_file = os.path.join(SPJ_DATA_PATH, "source_ids.bin")
                out_source_ids_mem = np.memmap(out_source_ids_file, dtype=np.uint16, mode="w+", shape=(sum(mem_len_list)))

            index = 0
            for i in range(len(datasets)):
                out_mem[index:index + mem_len_list[i]] = mem_data_list[i]
                if split == 'train':
                    out_source_ids_mem[index:index + mem_len_list[i]] = np.full(mem_len_list[i], i, dtype=np.uint16)
                index += mem_len_list[i]

            out_mem.flush()

    train_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    source_ids = np.memmap(
        os.path.join(SPJ_DATA_PATH, "source_ids.bin"), dtype=np.uint16, mode="r"
    ) if os.path.exists(os.path.join(SPJ_DATA_PATH, "source_ids.bin")) else None

    return {"train": train_data, "val": val_data, 'source_ids': source_ids}


def get_slimpajama_chunk1(num_proc=40):
    if not os.path.exists(os.path.join(SPJ_CHUNK_1_DATA_PATH, "train.bin")):
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        dataset = load_dataset("cerebras/SlimPajama-627B", split="train/chunk1")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    return {"train": train_data, "val": val_data}
