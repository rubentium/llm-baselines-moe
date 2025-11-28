import torch
import torch.nn as nn
import torch.nn.functional as F


def log_mean(x, dim):
    return torch.logsumexp(x, dim=dim) - torch.log(
        torch.tensor(x.shape[dim], dtype=torch.float32)
    )


def entropy_reg(logits: torch.Tensor, mean_over_batch: bool = True):
    """Entropy regularization for the router."""

    entropy_l = lambda l: -(l * l.exp()).sum(-1)
    # softmax over experts
    # logits: [batch_size * sequence_length, num_experts]
    logprobs = F.log_softmax(logits, dim=-1)
    if mean_over_batch:
        # take mean probability over batch
        logprobs = log_mean(logprobs, 0)

    return -entropy_l(logprobs).mean()


# two losses below are adapted from
# https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/architectures/moe/routing.py
def load_balancing_loss(logits: torch.Tensor, expert_indices: torch.Tensor) -> float:
    """Computes auxiliary load balancing loss as in Switch Transformer.

    See Switch Transformer (https://arxiv.org/abs/2101.03961). This function
    implements the loss function presented in equations (4) - (6). It aims to
    penalize those cases where the routing between experts is unbalanced.

    Args:
      logits: logits assigned to each expert per token. Shape:
        <float32>[batch_size * sequence_length, num_experts].
      expert_indices: <int>[batch_size * sequence_length, num_selected_experts]
        indices identifying the top num_selected_experts for a given token.

    Returns:
      The auxiliary loss.
    """
    # num_token = batch_size * sequence_length
    num_token, num_experts = logits.shape

    # Shape: [batch_size * sequence_length, num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_indices, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [batch_size * sequence_length, num_experts]
    expert_mask, _ = torch.max(expert_mask, dim=-2)

    # shape [num_experts]
    tokens_per_expert = torch.mean(expert_mask, dim=0, dtype=torch.float32)

    # compute router probability per expert in log space for numerical stability
    logprobs = F.log_softmax(logits, dim=-1)
    # take mean probability over batch
    # shape [num_experts]
    logprobs = log_mean(logprobs, dim=0)
    router_prob_per_expert = torch.exp(logprobs)
    return (
        torch.mean(  # mean over experts
            tokens_per_expert * router_prob_per_expert,
            dtype=torch.float32,
        )
        * num_experts
    )


def router_z_loss(router_logits: torch.Tensor) -> float:
    """Compute router z-loss.

     The router z-loss was introduced in Designing Effective Sparse Expert Models
     (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
     small in an effort to improve stability.

    Args:
      router_logits: <float>[batch_size * sequence_length, num_experts]
        router logits

    Returns:
      Scalar router z-loss.
    """
    num_tokens, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss, dtype=torch.float32) / (num_tokens)

def routing_div_loss(router_logits: torch.Tensor):
    """
    Router diversity loss as implemented in this paper: 
    https://arxiv.org/pdf/2505.22323
    """
    all_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32) # [num_tokens, num_experts]
    avg_expert_scores = torch.mean(all_probs, dim=0)  # [num_experts]
    pre_sum_terms = torch.square(all_probs - avg_expert_scores)  # [num_tokens, num_experts]
    diversity_loss =  -torch.mean(torch.mean(pre_sum_terms, dim=-1))  # scalar
    return diversity_loss

def expert_ortho_loss_old(token_matrix: torch.Tensor, input: torch.Tensor, chosen_experts: torch.Tensor):
    """
    Expert specialization loss as implemented in this paper: 
    https://arxiv.org/pdf/2505.22323

    tokens: tensor of shape [batch, seq_len] (token indices)
    input: tensor of shape [batch, seq_len, embed_dim]
    chosen_experts: tensor of shape [batch*seq_len, 1] (expert assigned per token)
    """
    device = input.device

    tokens_flat = token_matrix.view(-1).unsqueeze(-1)  # [batch*seq_len, 1]
    input_flat = input.view(-1, input.shape[-1])  # [batch*seq_len, embed_dim]
    
    norm_sq = input_flat.norm(p=2, dim=-1, keepdim=True) ** 2
    normed_input = input_flat / (norm_sq + 1e-6)    # [batch*seq_len, embed_dim]

    experts = torch.unique(chosen_experts)

    loss = torch.tensor(0.0).to(device=device)
    for e in experts:
        input_mask = (chosen_experts != e).bool()  # [unmasked, 1]
        normed_input_mask = (chosen_experts == e).bool()   # [masked, 1]
        pruned_input = input_flat[input_mask.squeeze(-1), :]   # [unmasked, embed_dim]
        pruned_normed_input = normed_input[normed_input_mask.squeeze(-1), :]    # [masked, embed_dim]
        dot_prod = pruned_input @ pruned_normed_input.T     # [unmasked, masked]    

        pruned_tokens_vert = tokens_flat[input_mask].unsqueeze(-1)  # [unmasked, 1]
        pruned_tokens_hor = tokens_flat[normed_input_mask].unsqueeze(-1)  # [masked, 1]
        token_mask = (pruned_tokens_vert == pruned_tokens_hor.T).bool()   # [unmasked, masked]

        loss += dot_prod[token_mask].sum()

    # dot_prod = input_flat @ normed_input.T  # [batch*seq_len, batch*seq_len]

    # loss = torch.tensor(0.0, device=input.device)
    # expert_mask = 1-(chosen_experts == chosen_experts.T).float()  # [batch*seq_len, batch*seq_len]
    # token_mask = (tokens_flat.unsqueeze(1) == tokens_flat.unsqueeze(0)).float()  # [batch*seq_len, batch*seq_len]
    # per_expert_dot_prod = dot_prod * token_mask * expert_mask
    # loss = torch.sum(per_expert_dot_prod)

    return loss


def expert_ortho_loss(tokens: torch.Tensor,
                                  input: torch.Tensor,
                                  chosen_experts: torch.Tensor,
                                  eps: float = 1e-6):
    """
    Orthogonality loss with denominator ||x_k||^2 as in the paper.

    Args:
        input_flat: [N, D] token embeddings
        tokens_flat: [N] token ids
        chosen_experts: [N] expert indices
        eps: small value for numerical stability
    Returns:
        scalar loss
    """
    input_flat = input.view(-1, input.shape[-1])
    tokens_flat = tokens.reshape(-1)
    x = input_flat  # keep original embeddings (not normalized)

    experts = torch.unique(chosen_experts)
    loss = x.new_tensor(0.0)

    for e in experts:
        mask_e = (chosen_experts == e).squeeze(-1)
        x_e = x[mask_e]               # [Ne, D]
        t_e = tokens_flat[mask_e]     # [Ne]

        mask_ne = ~mask_e
        x_ne = x[mask_ne]             # [Nne, D]
        t_ne = tokens_flat[mask_ne]   # [Nne]

        if x_e.size(0) == 0 or x_ne.size(0) == 0:
            continue

        dots = x_ne @ x_e.T           # [Nne, Ne]

        norm_sq = (x_e ** 2).sum(dim=-1)  # [Ne]

        token_mask = (t_ne.unsqueeze(1) == t_e.unsqueeze(0))  # [Nne, Ne]

        selected = dots[token_mask]
        denom = norm_sq.unsqueeze(0).expand_as(dots)[token_mask] + eps

        loss += (selected / denom).sum()

    return loss
