import torch
from opacus.accountants.analysis import rdp as privacy_analysis


DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 201)] + list(range(22, 500))


def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)


def get_privacy_spent_with_fixed_noise(
        sample_rate, num_steps, delta, sigma=1.0, alphas=None
):
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    noise_multiplier = sigma

    rdp = privacy_analysis.compute_rdp(
        q=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=num_steps,
        orders=alphas
    )

    eps, best_alpha = privacy_analysis.get_privacy_spent(
        orders=alphas, rdp=rdp, delta=delta
    )

    return float(eps), float(best_alpha)


def get_client_iter(
        sample_rate, max_eps, delta, sigma=1.0, alphas=None
):
    min_step = 0
    max_step = 1

    while get_privacy_spent_with_fixed_noise(
        sample_rate, max_eps, delta, sigma=sigma, alphas=alphas
    )[0] <= max_eps:
        min_step = max_step
        max_step = max_step * 2

    max_step = max_step - 1

    while min_step < max_step:
        mid_step = (min_step + max_step + 1) // 2
        eps_used, best_alpha = get_privacy_spent_with_fixed_noise(
            sample_rate, mid_step, delta, sigma=sigma, alphas=alphas
        )

        if max_eps < eps_used:
            max_step = mid_step - 1
        else:
            min_step = mid_step

    return min_step


def get_min_sigma(
        sample_rate, num_steps, delta, require_eps, alphas=None
):
    min_sigma = 0
    max_sigma = 1

    while get_privacy_spent_with_fixed_noise(
        sample_rate, num_steps, delta, sigma=max_sigma, alphas=alphas
    )[0] > require_eps:
        min_sigma = max_sigma
        max_sigma = max_sigma * 2

    while max_sigma-min_sigma > 1e-8:
        mid_sigma = (max_sigma+min_sigma) / 2
        eps, best_alpha = get_privacy_spent_with_fixed_noise(
            sample_rate, num_steps, delta, sigma=mid_sigma, alphas=alphas
        )

        if eps <= require_eps:
            max_sigma = mid_sigma
        else:
            min_sigma = mid_sigma

    return max_sigma
