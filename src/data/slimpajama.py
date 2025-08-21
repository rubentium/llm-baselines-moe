from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os 

BASE_PATH = "/mloscratch/homes/navasard/moe_doge/"
SPJ_DATA_PATH = os.path.join(BASE_PATH, "datasets/combined_slimpajama_debug_shuffled_full/")
SPJ_CHUNK_1_DATA_PATH = os.path.join(SPJ_DATA_PATH, "chunk1")


tknzr = tiktoken.get_encoding("gpt2")


def get_slimpajama_data(seq_len, batch_size, acc_steps, iterations, eval_freq, run_id=None, top_k=1, seed=1004):
    datasets_directory = "/mloscratch/homes/navasard/moe_doge/slimpajama"
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    if not os.path.exists(os.path.join(SPJ_DATA_PATH, "train.bin")):
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        datasets = [f for f in os.listdir(datasets_directory) 
                   if not os.path.isfile(os.path.join(datasets_directory, f))]
        splits = ['train', 'val']

        for split in splits:
            out_file = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            mem_data_list = []
            mem_len_list = []        
            
            # Load all datasets
            for i in range(len(datasets)):
                data = np.memmap(os.path.join(datasets_directory, datasets[i], f"{split}.bin"), 
                                dtype=np.uint16, mode="r")
                mem_data_list.append(data)
                mem_len_list.append(len(data))

            sequences = []
            source_ids = []
            
            for i in tqdm(range(len(datasets)), desc="Loading sequences"):
                data = mem_data_list[i]
                num_sequences = len(data) // seq_len
                
                for seq_idx in range(num_sequences):
                    start = seq_idx * seq_len
                    end = start + seq_len
                    sequences.append(data[start:end])
                    source_ids.append(i)  # Store source_id for each sequence
            
            # Create shuffle indices
            shuffle_indices = np.arange(len(sequences))
            np.random.shuffle(shuffle_indices)  # This uses the fixed seed
            
            # Write shuffled data
            total_tokens = len(sequences) * seq_len
            out_mem = np.memmap(out_file, dtype=np.uint16, mode="w+", shape=(total_tokens))
            out_source_ids_file = os.path.join(SPJ_DATA_PATH, f"{split}_source_ids.bin")
            out_source_ids_mem = np.memmap(out_source_ids_file, dtype=np.uint16, 
                                            mode="w+", shape=(total_tokens))
            
            for i in tqdm(range(len(sequences)), desc=f"Writing shuffled data - {split}"):
                # Get the shuffled index
                shuffled_idx = shuffle_indices[i]
                
                # Write the sequence and its corresponding source_id
                start = i * seq_len
                end = start + seq_len
                out_mem[start:end] = sequences[shuffled_idx]
                out_source_ids_mem[start:end] = np.full(seq_len, source_ids[shuffled_idx], dtype=np.uint16)

            out_mem.flush()
            out_source_ids_mem.flush()

    train_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    train_source_ids = np.memmap(
        os.path.join(SPJ_DATA_PATH, "train_source_ids.bin"), dtype=np.uint16, mode="r"
    ) if os.path.exists(os.path.join(SPJ_DATA_PATH, "train_source_ids.bin")) else None

    val_source_ids = np.memmap(
        os.path.join(SPJ_DATA_PATH, "val_source_ids.bin"), dtype=np.uint16, mode="r"
    ) if os.path.exists(os.path.join(SPJ_DATA_PATH, "val_source_ids.bin")) else None

    # this is for logging the expert assignments
    if run_id:
        assert top_k >= 1, "top_k must be at least 1"
        os.makedirs(os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}"), exist_ok=True)
        train_exprt_dir = os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "train.bin")
        val_exprt_dir = os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "val.bin")

        train_len = batch_size * acc_steps * seq_len * iterations # to save on memory only allocate memory for training token-expert assignments
        train_shape = (train_len, top_k) if top_k > 1 else (train_len,)

        assert iterations % eval_freq == 0, "Iterations must be divisible by eval_freq"
        max_num_batches = 24 # same as in eval in utils in optim folder
        val_len = int(batch_size * max_num_batches * seq_len * (iterations / eval_freq - 1))
        val_shape = (val_len*2, top_k) if top_k > 1 else (val_len*2,)

        # Create memmap for train and val expert assignments
        train_exprt_mem = np.memmap(train_exprt_dir, dtype=np.int8, mode="w+", shape=train_shape)
        train_exprt_mem[:] = np.full(train_shape, -1, dtype=np.int8)  # Initialize train with -1
        train_exprt_mem.flush()

        val_exprt_mem = np.memmap(val_exprt_dir, dtype=np.int8, mode="w+", shape=val_shape)
        val_exprt_mem[:] = np.full(val_shape, -1, dtype=np.int8)  # Initialize val with -1
        val_exprt_mem.flush()
        train_exprt_index, val_exprt_index = 0, 0

        # Create memmap for train and val loss
        train_loss = np.memmap(os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "train_loss.bin"), 
                                    dtype=np.float32, mode="w+", shape=train_shape)
        train_loss[:] = np.full(train_shape, -1.0, dtype=np.float32)  # Initialize train_loss with -1.0
        train_loss.flush()

        val_loss = np.memmap(os.path.join(BASE_PATH, f"expert_assignments/run_id_{run_id}", "val_loss.bin"), 
                                    dtype=np.float32, mode="w+", shape=val_shape)
        val_loss[:] = np.full(val_shape, -1.0, dtype=np.float32)  # Initialize val_loss with -1.0
        val_loss.flush()

        return {"train": train_data[:train_len+1],
                "val": val_data[:val_len+1], 
                "train_source_ids": train_source_ids, 
                "val_source_ids": val_source_ids,
                "train_exp": train_exprt_mem, 
                "val_exp": val_exprt_mem,
                "train_exp_index": train_exprt_index, 
                "val_exp_index": val_exprt_index,
                "train_loss": train_loss,
                "val_loss": val_loss}
    else:
        return {"train": train_data, 
                "val": val_data, 
                "train_source_ids": train_source_ids, 
                "val_source_ids": val_source_ids}


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
