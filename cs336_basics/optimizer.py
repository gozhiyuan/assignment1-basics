"""
Optimizer implementations for deep learning models.

This module contains custom implementations of popular optimizers:
- SGD: Stochastic Gradient Descent with learning rate scheduling
- AdamW: Adam optimizer with decoupled weight decay

The AdamW optimizer is particularly important as it fixes issues with the original
Adam optimizer regarding weight decay and L2 regularization.
"""

import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import reduce
from collections.abc import Callable, Iterable
 
import math

def cross_entropy(
    logits: Float[Tensor, "... vocab_size"], 
    targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    """
    Compute cross-entropy loss with numerical stability using einops.
    
    Args:
        logits: Predicted logits of shape (..., vocab_size) where ... represents batch dimensions
        targets: Target indices of shape (...) where ... represents batch dimensions
    
    Returns:
        Average cross-entropy loss across all batch dimensions
        
    The function implements the cross-entropy loss:
    ℓi = -log softmax(logits_i)[target_i]
    
    For numerical stability:
    1. Subtract the maximum logit from all logits
    2. Cancel out log and exp operations where possible
    3. Handle batch dimensions and return average
    """
    # Step 1: Subtract maximum for numerical stability
    # Compute max along vocab_size dimension and keep dims for broadcasting
    max_logits = reduce(logits, "... vocab_size -> ... 1", 'max')
    
    # Subtract max from all logits
    logits_stable = logits - max_logits
    
    # Step 2: Compute exp of stable logits
    exp_logits = torch.exp(logits_stable)
    
    # Step 3: Compute sum of exp for normalization
    sum_exp = reduce(exp_logits, "... vocab_size -> ... 1", 'sum')
    
    # Step 4: Compute log of softmax (canceling exp and log)
    # log(softmax(logits)[target]) = logits[target] - log(sum(exp(logits)))
    
    # Get the stable logits corresponding to targets
    # Use torch.gather since einops doesn't have gather function
    # 
    # torch.gather(input, dim, index) extracts values from input based on indices in index
    # 
    # Example with shapes:
    # logits_stable: [batch, seq, vocab_size] = [2, 3, 5]
    # targets: [batch, seq] = [2, 3] 
    # 
    # targets.unsqueeze(-1): [batch, seq, 1] = [2, 3, 1]
    # This adds a dimension to match the gather operation requirements
    # 
    # torch.gather(logits_stable, dim=-1, index=targets.unsqueeze(-1)):
    # - dim=-1: gather along the last dimension (vocab_size)
    # - For each position (batch_i, seq_j), extract the logit at index targets[batch_i, seq_j]
    # - Result shape: [batch, seq, 1] = [2, 3, 1]
    # 
    # .squeeze(-1): removes the last dimension, giving [batch, seq] = [2, 3]
    # 
    # Example with concrete values:
    # logits_stable[0, 0] = [0.1, 0.2, 0.3, 0.4, 0.5]  # vocab_size=5
    # targets[0, 0] = 2  # want the logit at index 2
    # result[0, 0] = 0.3  # extracted value
    target_logits_stable = torch.gather(logits_stable, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Compute log of sum of exp (using logsumexp trick)
    log_sum_exp = torch.log(sum_exp)
    
    # Compute cross-entropy loss: -logits_stable[target] + log_sum_exp
    cross_entropy_loss = -target_logits_stable + log_sum_exp.squeeze(-1)
    
    # Step 5: Return average across all batch dimensions
    return reduce(cross_entropy_loss, "... -> ", 'mean')


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer: Adam with Decoupled Weight Decay Regularization.
    
    AdamW is an improved version of Adam that fixes weight decay issues.
    The key insight is that L2 regularization and weight decay are NOT equivalent
    for adaptive gradient methods like Adam.
    
    Key features:
    1. Adaptive learning rates per parameter based on gradient statistics
    2. Momentum-like behavior through exponential moving averages
    3. Proper decoupled weight decay (applied directly to parameters)
    4. Bias correction for early training steps
    
    Algorithm overview:
    1. Compute exponential moving averages of gradients (m_t) and squared gradients (v_t)
    2. Apply bias correction to account for initialization at zero
    3. Update parameters using adaptive step size: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    4. Apply weight decay directly to parameters (decoupled from gradient)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (α) - controls overall step size (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square
               beta1 controls momentum (default: 0.9)
               beta2 controls scaling/adaptivity (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient for L2 regularization (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model and returns the loss
            
        Returns:
            The loss value if closure is provided, else None
        """
        loss = None if closure is None else closure()

        # Process each parameter group (allows different hyperparameters for different layers)
        for group in self.param_groups:
            # Extract hyperparameters for this group
            lr: float = group["lr"]                    # Learning rate (α)
            beta1, beta2 = group["betas"]              # Exponential decay rates
            eps: float = group["eps"]                  # Numerical stability constant
            weight_decay: float = group["weight_decay"] # L2 regularization strength

            # Process each parameter in the current group
            for p in group["params"]:
                # Skip parameters with no gradient (e.g., frozen layers)
                if p.grad is None:
                    continue
                    
                grad = p.grad.data  # Current gradient
                
                # AdamW doesn't support sparse gradients
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                # Get or initialize optimizer state for this parameter
                state = self.state[p]

                # === STATE INITIALIZATION ===
                # Initialize state on first call for this parameter
                if len(state) == 0:
                    state["step"] = 0                                    # Step counter
                    state["exp_avg"] = torch.zeros_like(p.data)         # First moment estimate (m_t)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)      # Second moment estimate (v_t)

                # Get state variables (these are references, so modifications persist)
                exp_avg = state["exp_avg"]        # m_t: exponential moving average of gradients
                exp_avg_sq = state["exp_avg_sq"]  # v_t: exponential moving average of squared gradients

                # Increment step counter
                state["step"] += 1
                step_t: int = state["step"]

                # === DECOUPLED WEIGHT DECAY ===
                # Apply weight decay directly to parameters (key difference from Adam)
                # This is "decoupled" because it's separate from the gradient-based update
                # Formula: θ_t = θ_t - λ * α * θ_t  (where λ = weight_decay, α = lr)
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # === MOMENT UPDATES ===
                # Update biased first moment estimate (exponential moving average of gradients)
                # Formula: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                # This provides momentum-like behavior, smoothing gradient updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment estimate (exponential moving average of squared gradients)
                # Formula: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                # This tracks the magnitude/scale of gradients for adaptive scaling
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # === BIAS CORRECTION ===
                # Correct for initialization bias (moments start at zero)
                # Without correction, early steps would be biased toward zero
                # Formula: m̂_t = m_t / (1 - β₁ᵗ), v̂_t = v_t / (1 - β₂ᵗ)
                bias_correction1 = 1.0 - beta1 ** step_t  # Correction for first moment
                bias_correction2 = 1.0 - beta2 ** step_t  # Correction for second moment

                # Compute effective step size with bias correction
                # This incorporates the bias correction into the learning rate
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                # === PARAMETER UPDATE ===
                # Compute adaptive denominator: √v̂_t + ε
                # The sqrt makes this RMSprop-like, ε prevents division by zero
                denom = exp_avg_sq.sqrt().add_(eps)
                
                # Final parameter update: θ_t = θ_{t-1} - step_size * m̂_t / (√v̂_t + ε)
                # addcdiv_ performs: p.data = p.data + value * (exp_avg / denom)
                # where value = -step_size (negative because we're minimizing)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # Linear warmup from 0 to max over [0, warmup_iters]
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    # Cosine decay from max to min over iterations [warmup_iters, cosine_cycle_iters]
    # Note: here, `cosine_cycle_iters` is interpreted as the total number of iterations
    # in the warmup + cosine cycle. Thus, the cosine phase length is
    # (cosine_cycle_iters - warmup_iters).
    if it < cosine_cycle_iters:
        progress = (it - warmup_iters) / max(1, (cosine_cycle_iters - warmup_iters))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine

    # After the cycle, stay at min
    return min_learning_rate


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Gradient clipping prevents exploding gradients by enforcing a maximum L2 norm on the 
    combined gradients across all parameters. This is crucial for training stability,
    especially in RNNs and deep networks.

    Algorithm:
    1. Compute the L2 norm of all gradients combined: ||g||₂
    2. If ||g||₂ ≤ max_l2_norm: do nothing (gradients are already small enough)
    3. If ||g||₂ > max_l2_norm: scale all gradients by factor M/(||g||₂ + ε)
       where M = max_l2_norm and ε = 1e-6 for numerical stability

    This ensures the resulting gradient norm is just under max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Numerical stability constant (PyTorch default)
    eps = 1e-6
    
    # Step 1: Collect all gradients and compute their combined L2 norm
    # We need to flatten all gradients into a single vector to compute the norm
    gradients = []
    for param in parameters:
        if param.grad is not None:
            # Flatten each parameter's gradient and add to list
            gradients.append(param.grad.view(-1))
    
    # If no gradients exist, nothing to clip
    if not gradients:
        return
    
    # Concatenate all gradients into a single 1D tensor
    all_gradients = torch.cat(gradients)
    
    # Compute the L2 norm of the combined gradient vector
    # ||g||₂ = sqrt(sum(gᵢ²)) for all gradient elements gᵢ
    total_norm = torch.norm(all_gradients, p=2)
    
    # Step 2: Check if clipping is needed
    if total_norm <= max_l2_norm:
        # Gradients are already within the limit, no clipping needed
        return
    
    # Step 3: Compute clipping factor
    # We want to scale gradients so that ||scaled_g||₂ = max_l2_norm
    # Scaling factor = max_l2_norm / (||g||₂ + ε)
    # The ε prevents division by zero and provides numerical stability
    clip_factor = max_l2_norm / (total_norm + eps)
    
    # Step 4: Apply clipping to all parameter gradients in-place
    for param in parameters:
        if param.grad is not None:
            # Scale each gradient by the clipping factor
            # This modifies param.grad in-place using mul_()
            param.grad.mul_(clip_factor)