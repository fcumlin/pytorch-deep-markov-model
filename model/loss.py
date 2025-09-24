import torch
import torch.nn as nn


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.zeros_like(mu1)

    return 0.5 * (
        logvar2 - logvar1 + (
            torch.exp(logvar1) + (mu1 - mu2).pow(2)
        ) / torch.exp(logvar2) - 1)


def gaussian_nll_loss(x_hat, x, Cw=None):
    """
    Negative log-likelihood for Gaussian emission: x ~ N(x_hat, Cw)
    
    Args:
        x_hat: predicted mean (batch_size, seq_len, obs_dim)
        x: true observations (batch_size, seq_len, obs_dim)  
        Cw: observation noise covariance (batch_size, obs_dim, obs_dim) or (batch_size, 1, obs_dim, obs_dim)
    """
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    
    batch_size, seq_len, obs_dim = x.shape
    
    if Cw is None:
        # Default to identity covariance
        Cw = torch.eye(obs_dim).unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
    
    if Cw.dim() == 4 and Cw.size(1) == 1:
        # Remove singleton time dimension
        Cw = Cw.squeeze(1)
    
    # Compute residuals
    residuals = x - x_hat  # (batch_size, seq_len, obs_dim)
    
    # Compute NLL for each time step
    nll = torch.zeros(batch_size, seq_len).to(x.device)
    
    for t in range(seq_len):
        residual_t = residuals[:, t, :]  # (batch_size, obs_dim)
        
        # Try to use Cholesky decomposition for efficiency
        try:
            L = torch.cholesky(Cw)  # (batch_size, obs_dim, obs_dim)
            # Solve L * y = residual_t for y
            y = torch.triangular_solve(residual_t.unsqueeze(-1), L, upper=False)[0].squeeze(-1)
            
            # NLL = 0.5 * (y^T * y + log|2π*Cw|)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
            nll[:, t] = 0.5 * (torch.sum(y**2, dim=-1) + log_det + obs_dim * torch.log(torch.tensor(2 * torch.pi)))
            
        except:
            # Fallback to standard computation if Cholesky fails
            try:
                Cw_inv = torch.inverse(Cw)
                _, log_det = torch.slogdet(Cw)
                
                quad_form = torch.sum(residual_t.unsqueeze(-2) @ Cw_inv @ residual_t.unsqueeze(-1), dim=(-2, -1))
                nll[:, t] = 0.5 * (quad_form.squeeze() + log_det + obs_dim * torch.log(torch.tensor(2 * torch.pi)))
            except:
                # Last resort: diagonal approximation
                if Cw.dim() == 3:
                    diag_cov = torch.diagonal(Cw, dim1=-2, dim2=-1)
                else:
                    diag_cov = Cw
                nll[:, t] = 0.5 * torch.sum((residual_t**2) / diag_cov + torch.log(2 * torch.pi * diag_cov), dim=-1)
    
    return nll


def dmm_loss_gaussian(x, x_hat, mu1, logvar1, mu2, logvar2, kl_annealing_factor=1, mask=None, Cws=None):
    """
    DMM loss for Gaussian emissions
    """
    kl_raw = kl_div(mu1, logvar1, mu2, logvar2)
    nll_raw = gaussian_nll_loss(x_hat, x, Cws)
    
    # Feature-dimension reduced KL (sum over latent dims)
    kl_fr = kl_raw.sum(dim=-1)  # (batch_size, seq_len)
    nll_fr = nll_raw  # Already (batch_size, seq_len)
    
    # Masking
    if mask is not None:
        mask = mask.gt(0).view(-1)
        kl_m = kl_fr.view(-1).masked_select(mask).mean()
        nll_m = nll_fr.view(-1).masked_select(mask).mean()
    else:
        kl_m = kl_fr.view(-1).mean()
        nll_m = nll_fr.view(-1).mean()

    loss = kl_m * kl_annealing_factor + nll_m

    return kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss


# Keep original function for backward compatibility
def dmm_loss(x, x_hat, mu1, logvar1, mu2, logvar2, kl_annealing_factor=1, mask=None, Cws=None):
    """Original DMM loss - routes to appropriate loss function"""
    if Cws is not None:
        return dmm_loss_gaussian(x, x_hat, mu1, logvar1, mu2, logvar2, kl_annealing_factor, mask, Cws)
    else:
        # Original Bernoulli loss
        return dmm_loss_bernoulli(x, x_hat, mu1, logvar1, mu2, logvar2, kl_annealing_factor, mask)


def dmm_loss_bernoulli(x, x_hat, mu1, logvar1, mu2, logvar2, kl_annealing_factor=1, mask=None):
    """Original Bernoulli DMM loss"""
    kl_raw = kl_div(mu1, logvar1, mu2, logvar2)
    nll_raw = nll_loss(x_hat, x)  # Original Bernoulli NLL
    
    kl_fr = kl_raw.mean(dim=-1)
    nll_fr = nll_raw.mean(dim=-1)
    
    if mask is not None:
        mask = mask.gt(0).view(-1)
        kl_m = kl_fr.view(-1).masked_select(mask).mean()
        nll_m = nll_fr.view(-1).masked_select(mask).mean()
    else:
        kl_m = kl_fr.view(-1).mean()
        nll_m = nll_fr.view(-1).mean()

    loss = kl_m * kl_annealing_factor + nll_m

    return kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss


def nll_loss(x_hat, x):
    """Original Bernoulli NLL loss"""
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)