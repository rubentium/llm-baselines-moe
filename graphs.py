import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

runs = ['20250805_092802']
num_tokens = 4980736000

for run_id in runs:
    assignment = np.memmap(
        f'/mloscratch/homes/navasard/moe_doge/expert_assignments/run_id_{run_id}/train.bin',
        dtype=np.int8, mode='r'
    )[:num_tokens]
    tokens = np.memmap(
        f'/mloscratch/homes/navasard/moe_doge/datasets/combined_slimpajama_debug_shuffled_full/train.bin',
        dtype=np.uint16, mode='r'
    )[:num_tokens]
    source_ids = np.memmap(
        f'/mloscratch/homes/navasard/moe_doge/datasets/combined_slimpajama_debug_shuffled_full/train_source_ids.bin',
        dtype=np.int32, mode='r'
    )[:num_tokens]
    losses = np.memmap(
        f'/mloscratch/homes/navasard/moe_doge/expert_assignments/run_id_{run_id}/train_loss.bin',
        dtype=np.float32, mode='r'
    )[:num_tokens]
    assert len(tokens) == len(source_ids) == len(assignment) == len(losses), "Mismatch in lengths of tokens, source ids, assignment, and losses."

    num_clusters = 7
    token_batch_size = 512 * 32
    group_size = 800
    experts = range(num_clusters)

    num_batches = (len(assignment) - token_batch_size + 1) // token_batch_size
    loss_per_expert_per_batch = []

    for batch_idx in range(num_batches):
        start = batch_idx * token_batch_size
        end = start + token_batch_size

        batch_assign = assignment[start:end]
        batch_loss = losses[start:end]

        expert_losses = {}
        for expert in experts:
            mask = batch_assign == expert
            if np.any(mask):
                expert_losses[expert] = np.mean(batch_loss[mask])
        
        loss_per_expert_per_batch.append(expert_losses)

    tokens_aligned = tokens[:num_tokens]
    source_ids_aligned = source_ids[:num_tokens]

    experts = sorted(set(assignment))
    sources = sorted(set(source_ids_aligned))

    # --- Cluster counts (with duplicates) ---
    clusters = defaultdict(Counter)

    # --- Bidirectional counts ---
    expert_source_counts = {e: Counter() for e in experts}  # expert → source
    source_expert_counts = {s: Counter() for s in sources}  # source → expert

    # --- Loop over token assignments ---
    for i in range(num_tokens):
        token = tokens_aligned[i]
        expert = assignment[i]
        source = source_ids_aligned[i]

        # Cluster: count every token assignment (with repeats)
        clusters[expert][token] += 1

        # Bidirectional tallies
        expert_source_counts[expert][source] += 1
        source_expert_counts[source][expert] += 1

    # --- Write output to file instead of printing ---
    output_filename = f'/mloscratch/homes/navasard/moe_doge/llm-baselines-moe/graphs/{run_id}-grouped-dedublicated.txt'
    with open(output_filename, 'w') as f:
        # --- Write cluster sizes ---
        f.write("=== Cluster Sizes (Total Tokens per Expert, including duplicates) ===\n")
        for expert in experts:
            total = sum(clusters[expert].values())
            f.write(f"Expert {expert:2d}: {total:,} tokens\n")

        # --- Expert-centric view: P(source | expert) ---
        f.write("\n=== P(Source | Expert): Where each expert's tokens came from ===\n")
        for expert in experts:
            total = sum(expert_source_counts[expert].values())
            f.write(f"Expert {expert:2d} (total {total:,} tokens):\n")
            for source in sources:
                count = expert_source_counts[expert][source]
                pct = 100 * count / total if total else 0.0
                f.write(f"  From Source {source:2d}: {pct:6.2f}% ({count:,})\n")

        # --- Source-centric view: P(expert | source) ---
        f.write("\n=== P(Expert | Source): Where each source's tokens were routed ===\n")
        for source in sources:
            total = sum(source_expert_counts[source].values())
            f.write(f"Source {source:2d} (total {total:,} tokens):\n")
            for expert in experts:
                count = source_expert_counts[source][expert]
                pct = 100 * count / total if total else 0.0
                f.write(f"  To Expert {expert:2d}: {pct:6.2f}% ({count:,})\n")

    # --- Aggregating over group of batches (e.g., 800 at a time) ---
    num_groups = (len(loss_per_expert_per_batch) + group_size - 1) // group_size
    loss_matrix_grouped = np.full((num_clusters, num_groups), np.nan)

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min((group_idx + 1) * group_size, len(loss_per_expert_per_batch))
        group = loss_per_expert_per_batch[start:end]

        # Collect per-expert losses in this group
        expert_sums = {i: 0.0 for i in experts}
        expert_counts = {i: 0 for i in experts}
        
        for batch in group:
            for expert, loss in batch.items():
                expert_sums[expert] += loss
                expert_counts[expert] += 1
        
        for expert in experts:
            if expert_counts[expert] > 0:
                loss_matrix_grouped[expert, group_idx] = expert_sums[expert] / expert_counts[expert]

    # --- Plotting (with grouped batch losses) ---
    plt.figure(figsize=(14, 7))
    batches_grouped = range(num_groups)

    for expert in experts:
        plt.plot(
            batches_grouped,
            loss_matrix_grouped[expert],
            solid_capstyle='round',
            linewidth=2,
            label=f'Expert {expert}'
        )

    plt.title(f"Expert Losses Across {group_size}-Batch Groups")
    plt.xlabel(f"Group Index (each group = {group_size} batches)")
    plt.ylabel("Average Loss")
    plt.legend(title="Experts", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)

    # --- Inset cluster size bar chart (moved to top-right corner) ---
    ax2 = plt.gca().inset_axes([0.65, 0.55, 0.3, 0.35])  # [x0, y0, width, height]
    cluster_sizes = [len(clusters[c]) for c in sorted(clusters)]
    ax2.bar(experts, cluster_sizes, color='tab:gray')
    ax2.set_title("Final Token Cluster Sizes", fontsize=10)
    ax2.set_xticks(experts)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig(f'/mloscratch/homes/navasard/moe_doge/llm-baselines-moe/graphs/{run_id}-grouped-dedublicated.png')