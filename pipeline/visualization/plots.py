"""Publication-quality visualization for TSRM imputation results."""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

COLORS = {
    'original': '#1f77b4',
    'masked': '#7f7f7f',
    'imputed': '#ff7f0e',
    'locf': '#d62728',
    'linear': '#9467bd',
}


def plot_imputation_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    imputed: np.ndarray,
    variable_name: str,
    time_axis: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot time series comparison of original, masked, and imputed data.

    Args:
        original: Ground truth values [n_steps] or [n_samples, n_steps]
        masked: Masked data with NaN at missing positions
        imputed: Imputed values
        variable_name: Name of variable for title/labels
        time_axis: Time values for x-axis (optional)
        save_path: Path to save figure (optional)
        title: Custom title (optional)

    Returns:
        matplotlib Figure
    """
    # Handle multi-dimensional input
    if original.ndim == 2:
        original = original[0, :]
    if masked.ndim == 2:
        masked = masked[0, :]
    if imputed.ndim == 2:
        imputed = imputed[0, :]

    n_steps = len(original)
    if time_axis is None:
        time_axis = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(time_axis, original, color=COLORS['original'],
            label='Ground Truth', linewidth=1.5, alpha=0.8)

    ax.plot(time_axis, masked, color=COLORS['masked'],
            label='Observed', linewidth=1, marker='o', markersize=3, alpha=0.6)

    ax.plot(time_axis, imputed, color=COLORS['imputed'],
            label='Imputed', linewidth=1.5, linestyle='--')

    imputed_mask = np.isnan(masked) & np.isfinite(imputed)
    if imputed_mask.any():
        ax.fill_between(time_axis,
                        np.nanmin(original), np.nanmax(original),
                        where=imputed_mask,
                        alpha=0.2, color=COLORS['imputed'],
                        label='Imputed Region')

    ax.set_xlabel('Time Step')
    ax.set_ylabel(variable_name)
    ax.set_title(title or f'{variable_name} - Imputation Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_skill_scores(
    results: Dict,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot grouped bar chart of skill scores.

    Args:
        results: Dict with structure:
            {'skill_vs_locf': {'msess': float, 'maess': float},
             'skill_vs_linear': {'msess': float, 'maess': float}}
            OR
            {'per_variable': {var_name: {'msess': {baseline: score}, 'maess': {...}}}}
        save_path: Path to save figure
        title: Custom title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Check if results have per_variable structure
    if 'per_variable' in results:
        per_var = results['per_variable']
        if not per_var:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight')
            return fig

        variables = list(per_var.keys())
        baselines = ['locf', 'linear']

        x = np.arange(len(variables))
        width = 0.35

        for i, baseline in enumerate(baselines):
            scores = []
            for var in variables:
                var_data = per_var.get(var, {})
                msess = var_data.get('msess', {}).get(baseline, 0)
                scores.append(msess)

            _ = ax.bar(x + i * width, scores, width,
                          label=f'vs {baseline.upper()}',
                          color=COLORS.get(baseline, f'C{i}'))

        ax.set_xlabel('Variable')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(variables, rotation=45, ha='right')

    else:
        # Simple structure with skill_vs_locf and skill_vs_linear
        methods = ['vs LOCF', 'vs Linear']
        msess_values = [
            results.get('skill_vs_locf', {}).get('msess', 0),
            results.get('skill_vs_linear', {}).get('msess', 0)
        ]
        maess_values = [
            results.get('skill_vs_locf', {}).get('maess', 0),
            results.get('skill_vs_linear', {}).get('maess', 0)
        ]

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width/2, msess_values, width, label='MSESS', color='steelblue')
        ax.bar(x + width/2, maess_values, width, label='MAESS', color='coral')

        ax.set_xticks(x)
        ax.set_xticklabels(methods)

    ax.set_ylabel('Skill Score')
    ax.set_title(title or 'TSRM Skill Scores vs Baselines')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def plot_mse_heatmap(
    results: Dict,
    metric: str = 'mse',
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot heatmap of MSE across variables.

    Args:
        results: Results dict with per_variable data
            e.g., {'per_variable': {var_name: {'mse': float, 'mae': float}}}
            OR simple structure {var_name: {'mse': float, 'mae': float}}
        metric: Metric name ('mse' or 'mae')
        save_path: Path to save figure
        title: Custom title

    Returns:
        matplotlib Figure
    """
    # Determine structure
    if 'per_variable' in results:
        per_var = results['per_variable']
    else:
        per_var = results

    if not per_var:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    variables = list(per_var.keys())
    values = [per_var.get(v, {}).get(metric, np.nan) for v in variables]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Normalize for color mapping
    values_array = np.array(values).reshape(1, -1)
    im = ax.imshow(values_array, aspect='auto', cmap='RdYlGn_r')

    ax.set_xticks(np.arange(len(variables)))
    ax.set_xticklabels([v.replace('GDC_', '') for v in variables], rotation=45, ha='right')
    ax.set_yticks([])

    # Add values
    for i, v in enumerate(variables):
        val = values[i]
        if not np.isnan(val):
            ax.text(i, 0, f'{val:.4f}', ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, label=metric.upper())

    ax.set_title(title or f'{metric.upper()} per Variable')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

    return fig
