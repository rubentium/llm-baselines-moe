import numpy as np

train = np.memmap('/mloscratch/homes/navasard/moe_doge/llm-baselines-moe/src/data/expert_assignments/run_id_20250725_142419/train.bin', dtype=np.int8, mode='r')
val = np.memmap('/mloscratch/homes/navasard/moe_doge/llm-baselines-moe/src/data/expert_assignments/run_id_20250725_142419/val.bin', dtype=np.int8, mode='r')

# for i in range(len(val)):
#     if val[i] == -1:
#         print(i)
print(train[-1000:])
print(val[-1000:])

print(len(train), len(val))