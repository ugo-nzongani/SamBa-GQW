import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches

def distribution_quality(prob,quality):
    """
    Parameters:
    - prob: Array containing the probabilities of quantum states measurements.
    - quality: Array containing the quality of the solutions.
    
    Returns:
    - The quality of the distribution described by `prob`.
    """
    return np.dot(prob,quality)

def rank_solutions_by_quality(quality_dict):
    """
    Parameters:
    - quality_dict: Dictionnary whose keys are integer representing quantum states and values are the associated quality.
    
    Returns:
    - Dictionnary whose keys are the rank and the values are list of integer representing the states.
    """
    quality_arr = np.array(list(quality_dict.values()))
    key_arr = np.array(list(quality_dict.keys()))

    # Get unique qualities and their inverse index mapping
    unique_qualities, inverse = np.unique(quality_arr, return_inverse=True)
    sorted_indices = np.argsort(-unique_qualities)  # descending

    # Map rank to list of keys
    rank_dict = {}
    for rank, q_idx in enumerate(sorted_indices):
        matching_keys = key_arr[inverse == q_idx].tolist()
        rank_dict[rank] = matching_keys

    return rank_dict

def probability_of_rank(rank_dict, r, prob):
    """
    Parameters:
    - rank_dict: dict of {rank: list of indices}.
    - r: target rank (int).
    - prob: numpy array of probabilities.
    
    Returns:
    - Total probability of measuring a state with rank r.
    """
    if r not in rank_dict:
        raise ValueError(f"Rank {r} not found in rank_dict.")
    
    indices = rank_dict[r]
    
    if any(i >= len(prob) or i < 0 for i in indices):
        raise IndexError("One or more indices in rank_dict are out of bounds for the probability array.")
    
    return np.sum(prob[indices])

def plot_rank_probabilities(rank_list, prob_per_rank, title='', save=False, figname='rank_prob.png', errors=None, threshold=-1):
    """
    Plots the probability of measuring each rank based on the rank_list and prob_per_rank array.
    
    Parameters:
    - rank_list: list of ranks (x-axis).
    - prob_per_rank: list of probabilities per rank (y-axis).
    - title: optional title for the plot.
    - save: whether to save the plot.
    - figname: filename to save the plot.
    - errors: List of error bars.
    - threshold: minimum probability required to be displayed on the plot.
    """
    filtered_ranks = []
    filtered_probs = []
    filtered_errors = []

    if errors is not None:
        for r, p, e in zip(rank_list, prob_per_rank, errors):
            if p > threshold:
                filtered_ranks.append(r)
                filtered_probs.append(p)
                filtered_errors.append(e)
    else:
        for r, p in zip(rank_list, prob_per_rank):
            if p > threshold:
                filtered_ranks.append(r)
                filtered_probs.append(p)

    x_label = np.arange(len(filtered_ranks))

    if errors is None:
        plt.bar(x_label, filtered_probs, color='skyblue')
    else:
        plt.bar(x_label, filtered_probs, yerr=filtered_errors, capsize=5, color='skyblue')

    if threshold > 0.:
        plt.plot([], [], ' ', label=f'Threshold = {threshold}')
        plt.legend(frameon=False)
        
    plt.xlabel('Rank')
    plt.ylabel('Probability')

    if title:
        plt.title(title)

    plt.xticks(ticks=x_label, labels=filtered_ranks)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save:
        plt.savefig(figname, dpi=300)
        
    plt.show()

def rank_mean_errors(prob_per_rank_list):
    max_len = max(len(sublist) for sublist in prob_per_rank_list)
    num_instances = len(prob_per_rank_list)

    padded = np.full((num_instances, max_len), np.nan, dtype=np.float32)

    for idx, sublist in enumerate(prob_per_rank_list):
        padded[idx, :len(sublist)] = sublist

    means = np.nanmean(padded, axis=0).tolist()
    errors = np.nanstd(padded, axis=0, ddof=1).tolist()

    # Replace nan values in errors where there was only one observation
    for i in range(max_len):
        valid_count = np.sum(~np.isnan(padded[:, i]))
        if valid_count <= 1:
            errors[i] = 0.0

    return means, errors


def compute_mean_and_error_for_ranks(data, rank_list=None, error_type='sem'):
    max_ranks = max(len(instance) for instance in data)
    num_instances = len(data)

    if rank_list is None or rank_list == []:
        rank_list = list(range(max_ranks))

    mean_data = []
    error_data = []

    for rank_idx in rank_list:
        rank_series_list = [instance[rank_idx] for instance in data if len(instance) > rank_idx]

        if not rank_series_list:
            continue

        max_time = max(len(series) for series in rank_series_list)
        num_series = len(rank_series_list)

        padded = np.full((num_series, max_time), np.nan, dtype=np.float32)
        for idx, series in enumerate(rank_series_list):
            padded[idx, :len(series)] = series

        mean_series = np.nanmean(padded, axis=0).tolist()

        if error_type == 'sem':
            error_series = sem(padded, axis=0, nan_policy='omit').tolist()
        else:
            error_series = np.nanstd(padded, axis=0, ddof=1).tolist()

        # Ensure error is 0 where there was only one observation
        for i in range(max_time):
            valid_count = np.sum(~np.isnan(padded[:, i]))
            if valid_count <= 1:
                error_series[i] = 0.0

        mean_data.append(mean_series)
        error_data.append(error_series)

    return mean_data, error_data

def compute_mean_and_error_over_time(data, error_type='sem'):
    max_time = max(len(seq) for seq in data)
    num_sequences = len(data)

    padded = np.full((num_sequences, max_time), np.nan, dtype=np.float32)
    for idx, seq in enumerate(data):
        padded[idx, :len(seq)] = seq

    mean_values = np.nanmean(padded, axis=0).tolist()

    if error_type == 'sem':
        error_values = sem(padded, axis=0, nan_policy='omit').tolist()
    else:
        error_values = np.nanstd(padded, axis=0, ddof=1).tolist()

    for i in range(max_time):
        valid_count = np.sum(~np.isnan(padded[:, i]))
        if valid_count <= 1:
            error_values[i] = 0.0

    return mean_values, error_values

def plot_performance(distribution_quality_over_time,
                     participation_ratio_over_time,
                     probabilities_per_rank_over_time,
                     times,
                     quality_list,
                     rank_list,
                     rank_dict,
                     pourcentage=0,
                     error_type='sem',
                     distrib_error=None,
                     participation_error=None,
                     prob_per_rank_error=None,
                     save=False,
                     figname='perf.png',
                     marker_list=None,
                     markevery=1,
                     markersize=5,
                     show_colorbar=True,
                     show_legend=True,
                     fontsize_labels=12,
                     fontsize_data=10,
                     fontsize_legend=10,
                     legend_pos=(0,0),
                     xlabel='$t$'
):
    
    fig, ax = plt.subplots()
    
    # courbe distribution quality
    plt.plot(times, distribution_quality_over_time, 
             label='Distribution quality', color='orange', linestyle='--')
    if distrib_error is not None:
        plt.fill_between(times,
                         [m - e for m, e in zip(distribution_quality_over_time, distrib_error)],
                         [m + e for m, e in zip(distribution_quality_over_time, distrib_error)],
                         alpha=0.3, color='orange')
    
    # courbe participation ratio
    plt.plot(times, participation_ratio_over_time, 
             label='Participation ratio', color='lime', linestyle='dashdot')
    if participation_error is not None:
        plt.fill_between(times,
                         [m - e for m, e in zip(participation_ratio_over_time, participation_error)],
                         [m + e for m, e in zip(participation_ratio_over_time, participation_error)],
                         alpha=0.3, color='lime')
        
    if marker_list is None:
        marker_list = ["o","s","v","x","^","<",">","*","h","D","d","|","p",
           ".", ",", "1", "2", "3", "4", "8", "P", "H", "+", "X", "_"]

    total_ranks = len(probabilities_per_rank_over_time)
    all_ranks = set(range(total_ranks))
    individual_ranks = set(rank_list)
    other_ranks = list(all_ranks - individual_ranks)

    # Setup colormap
    # Setup colormap
    if show_colorbar:
        quality_array = np.array(quality_list)[rank_list]

        if np.min(quality_array) == np.max(quality_array):
            # Only one unique value: fix the range manually
            norm = colors.Normalize(vmin=0, vmax=1)
        else:
            norm = colors.Normalize(vmin=np.min(quality_array), vmax=np.max(quality_array))

        cmap = cm.cool
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    
    # Plot individual ranks (sans labels pour la légende)
    for k, plot_rank_idx in enumerate(rank_list):
        mean_series = np.array(probabilities_per_rank_over_time[plot_rank_idx])
        err_series = (np.array(prob_per_rank_error[plot_rank_idx])
                      if prob_per_rank_error is not None else None)

        color = cmap(norm(quality_list[plot_rank_idx])) if show_colorbar else None

        line, = ax.plot(times, mean_series,
                        marker=marker_list[k],
                        markersize=markersize,
                        markerfacecolor='none',
                        markevery=markevery,
                        linestyle='-',
                        color=color,
                        alpha=1.)
        
        if err_series is not None:
            ax.fill_between(times,
                            mean_series - err_series,
                            mean_series + err_series,
                            alpha=0.2,
                            color=line.get_color())
    
    # Courbes "autres ranks"
    if other_ranks and pourcentage > 0:
        def top_percent(all_ranks, pourcentage):
            n = int(len(all_ranks) * pourcentage / 100)
            return all_ranks[:n]
        best_ranks = top_percent(list(rank_dict.keys()),pourcentage)
        #print(best_ranks)
        others_mean = np.sum([np.array(probabilities_per_rank_over_time[r]) for r in best_ranks], axis=0)

        ax.plot(times, others_mean,
                label="Probability "+str(pourcentage)+"\% best rankings",
                linestyle='dotted',
                color='gray')

    # Axes
    ax.tick_params(axis='x', labelsize=fontsize_data)
    ax.tick_params(axis='y', labelsize=fontsize_data)
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)

    # === Légende ===
        # === Legend ===
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()

        max_per_line = 4
        n_ranks = len(rank_list)

        # Split rank_list into chunks of size <= 4
        for chunk_start in range(0, n_ranks, max_per_line):
            chunk = rank_list[chunk_start:chunk_start + max_per_line]

            # Build the markers for this chunk
            combined_handles = tuple(
                Line2D([0], [0],
                       color=(cmap(norm(quality_list[r])) if show_colorbar else "black"),
                       linestyle="-",
                       marker=marker_list[i % len(marker_list)],
                       markerfacecolor="none",
                       markersize=markersize)
                for i, r in enumerate(chunk, start=chunk_start)
            )

            # Build the label string
            labels_str = ", ".join(str(r) for r in chunk)

            handles.append(combined_handles)
            labels.append(f"Probability rankings ({labels_str})")

        ax.legend(
            handles, labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower left",
            bbox_to_anchor=legend_pos,
            fontsize=fontsize_legend,
            frameon=True
        )
        plt.subplots_adjust(bottom=0.25)


    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Ranking quality", fontsize=fontsize_labels)
        cbar.ax.tick_params(labelsize=fontsize_data)

    plt.tight_layout()

    if save:
        plt.savefig(figname,dpi=300)

    plt.show()

def plot_ranking_distribution(prob_per_rank_final,
                              prob_per_rank_init=None,
                              threshold=1e-3,
                              num_ticks=15,
                              inset_plot_size=50,
                              inset_plot_label='Initial ranking distribution',
                              main_plot_label='Final ranking distribution',
                              save=False,
                              figname='ranking_distribution.png'
                              ):


    # Filter the data for the main plot
    filtered_indices = np.where(prob_per_rank_final > threshold)[0]
    filtered_values = prob_per_rank_final[filtered_indices]
    filtered_labels = filtered_indices
    filtered_x = np.arange(len(filtered_indices))

    # Create figure and main axis
    fig, ax = plt.subplots()
    bar_final = ax.bar(filtered_x, filtered_values, color='blue', alpha=0.8)
    ax.set_xlabel("Ranking")
    ax.set_ylabel("Probability")

    # Contrôle de la densité des ticks
    tick_indices = np.linspace(0, len(filtered_labels) - 1, num_ticks, dtype=int)
    tick_positions = filtered_x[tick_indices]
    tick_labels = [filtered_labels[i] for i in tick_indices]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x')

    # Inset axis
    if prob_per_rank_init is not None:
        ax_inset = inset_axes(ax, width=f"{inset_plot_size}%", height=f"{inset_plot_size}%", loc='upper right',bbox_to_anchor=(-0.00, 0.0005, 1, 1),  # (x0, y0, width, height)
            bbox_transform=ax.transAxes)
        x_inset = np.arange(len(prob_per_rank_init))
        bar_init = ax_inset.bar(x_inset, prob_per_rank_init, width=1., color='magenta', label='Initial ranking distribution', alpha=1)
        for bar in bar_init:
            bar.set_rasterized(True)
        # Compute 5 meaningful x-ticks
        size = len(x_inset)
        tick_positions = [0, size // 4, size // 2, 3 * size // 4, size - 1]
        tick_labels = [str(i) for i in tick_positions]

        ax_inset.set_xticks(tick_positions)
        ax_inset.set_xticklabels(tick_labels)
        ax_inset.tick_params(axis='y')
        ax.tick_params(axis='y')

    # Shared legend outside the plot (you can also put it inside with `loc=...`)
    final_patch = mpatches.Patch(color='blue', alpha=1, label=main_plot_label)
    init_patch = mpatches.Patch(color='magenta', alpha=1, label=inset_plot_label)

    if prob_per_rank_init is None:
        handles=[final_patch]
    else:
        handles=[init_patch, final_patch]
    ax.legend(handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.66, 0.4),  # below the plot
            ncol=1, frameon=True)
    
    if save:
        plt.savefig(figname)

    plt.show()