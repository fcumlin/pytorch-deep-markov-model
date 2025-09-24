import torch
import torch.nn as nn
import logging
from model.metric import _wasserstein_distance_gaussian


class _TransitionModelWrapper(nn.Module):

    def __init__(self, dmm_model):
        super().__init__()
        self.transition = dmm_model.transition
        
    def forward(self, prev_state):
        with torch.no_grad():
            mu, logvar = self.transition(prev_state)
            pred_cov = torch.diag_embed(torch.exp(logvar))
            
        return mu, pred_cov


def test_transition_model(dmm_model, test_dataloader, device):
    """
    Test only the transition model component of DMM.
    
    Args:
        dmm_model: Trained DeepMarkovModel
        test_dataloader: DataLoader with (prev_state, next_state, mean_next_state, cov_next_state)
        device: torch device
    """
    total_log_likelihood = 0.0
    total_kl_divergence = 0.0
    total_wasserstein = 0.0
    mse = 0.0
    total_samples = 0
    
    # Wrap the transition model
    markov_state_model = _TransitionModelWrapper(dmm_model)
    markov_state_model.eval()
    
    for i, batch in enumerate(test_dataloader):
        prev_state, next_state, mean_next_state, cov_next_state = batch
        prev_state = prev_state.to(device=device)
        
        # Get model predictions from transition model only
        with torch.no_grad():
            pred_mean, pred_cov = markov_state_model(prev_state)
        
        pred_mean = pred_mean.cpu()
        pred_cov = pred_cov.cpu()
        
        # Create predicted distribution
        try:
            pred_dist = torch.distributions.MultivariateNormal(
                pred_mean, pred_cov
            )
            log_likelihood = pred_dist.log_prob(next_state)
        except:
            # Fallback to diagonal if covariance is ill-conditioned
            pred_cov_diag = torch.diagonal(pred_cov, dim1=-2, dim2=-1)
            pred_dist = torch.distributions.MultivariateNormal(
                pred_mean, torch.diag_embed(pred_cov_diag)
            )
            log_likelihood = pred_dist.log_prob(next_state)
        
        # Only compute KL divergence and Wasserstein if true distribution is available
        if isinstance(mean_next_state, torch.Tensor):
            try:
                true_dist = torch.distributions.MultivariateNormal(
                    mean_next_state, cov_next_state
                )
                kl_div = torch.distributions.kl_divergence(pred_dist, true_dist)
                wasserstein = _wasserstein_distance_gaussian(
                    pred_mean, pred_cov, mean_next_state, cov_next_state
                )
                
                total_kl_divergence += kl_div.sum().item()
                total_wasserstein += wasserstein.sum().item()
            except:
                # Skip this batch if distributions are ill-conditioned
                logging.warning(f"Skipping batch {i} due to ill-conditioned distributions")
                continue
        
        total_log_likelihood += log_likelihood.sum().item()
        mse += torch.mean((pred_mean - next_state) ** 2).item()
        total_samples += prev_state.shape[0]
    
    # Log results
    logging.info('----------------------------')
    logging.info(f'Test MSE: {mse / total_samples}')
    logging.info(f'Test log likelihood: {total_log_likelihood / total_samples}')
    
    # Only log KL and Wasserstein if we computed them
    if isinstance(mean_next_state, torch.Tensor):
        logging.info(f'Test KL divergence: {total_kl_divergence / total_samples}')
        logging.info(f'Test Wasserstein distance: {total_wasserstein / total_samples}')
    else:
        logging.info('KL divergence and Wasserstein distance not computed (no target distribution)')
        total_kl_divergence = None
        total_wasserstein = None
    
    logging.info('----------------------------')
    
    return (total_log_likelihood / total_samples, 
            total_kl_divergence / total_samples if total_kl_divergence is not None else None)
