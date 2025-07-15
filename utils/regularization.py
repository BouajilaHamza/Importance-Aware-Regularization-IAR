import torch

def attentive_l2_regularization(weights, attention_scores, lambda_reg):
    # L_reg = lambda * sum_i (1 - a_i) * w_i^2
    # Ensure weights and attention_scores are flattened and have the same shape
    weights_flat = weights.view(-1)
    attention_scores_flat = attention_scores.view(-1)

    if weights_flat.shape != attention_scores_flat.shape:
        raise ValueError("Weights and attention scores must have the same shape after flattening.")

    regularization_term = torch.sum((1 - attention_scores_flat) * (weights_flat ** 2))
    return lambda_reg * regularization_term

