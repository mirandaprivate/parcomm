"""Plot the Fig. 1 in the paper.
"""
import argparse
import csv
import math
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


PIC_NAME = 'experiment.png'
VAR_NUM_NAME = 'var_num.csv'
VAR_SIZE_NAME = 'var_size.csv'

FIG_SIZE_SINGLE_COLUMN = (10, 5)
LABEL_SIZE_SINGLE_COLUMN = 14
# TITLE_SIZE_SINGLE_COLUMN = 16
TITLE_SIZE_SINGLE_COLUMN = 16
TICK_SIZE = 12
SPLINE_WIDTH = 1
LEGEND_FONT_SIZE = 14

# FIG_SIZE_DOUBLE_COLUMN = (16, 4)
# LABEL_SIZE_DOUBLE_COLUMN = 10
# TITLE_SIZE_DOUBLE_COLUMN = 12

LABEL_SIZE_3D = 14
TITLE_SIZE_3D = 16

LINE_STYLE = '-'
LINE_WIDTH = 1.


COLOR_PALETTE = ["#ea5545", "#f46a9b", "#ef9b20", "#edbf33", "#ede15b",
                 "#bdcf32", "#87bc45", "#00bfa0", "#27aeef", "#6481D9"]
MARKER_STYLE = "s"
MARKER_SIZE = 4
MARKER_SIZE_BIG = 30

Y_SUBTITLE = -0.4

COLOR = COLOR_PALETTE[0]
COLOR2 = COLOR_PALETTE[8]
LINE_STYLE = ':'
LINE_WIDTH = 0.85

my_cmap = sns.diverging_palette(120., 255., s=100, as_cmap=True) if sns else None

def plot_scatter():
    """The scatter plot for the RIM results and the closed-form solutions.
    """

    n_label = []
    n_series = []
    num_label = []
    num_series = []

    depth_label = []
    pars_label = []

    var_num_commit_time = []
    var_num_prover_time = []
    var_num_proof_size = []
    var_num_verifier_time = []

    var_size_commit_time = []
    var_size_prover_time = []
    var_size_proof_size = []
    var_size_verifier_time = []

    var_num_label = []
    var_num_series = []
    var_size_label = []
    var_size_series = []

    def build_ticks(series_groups, label_groups):
        """Return ordered unique tick positions/labels from multiple series."""
        ticks = []
        tick_labels = []
        seen = set()
        for series, labels in zip(series_groups, label_groups):
            for value, label in zip(series, labels):
                if value in seen:
                    continue
                seen.add(value)
                ticks.append(value)
                tick_labels.append(label)
        return ticks, tick_labels

    def highlight_last_point(ax, x_data, y_data, color):
        """Mark the final data point with a hollow square."""
        if not x_data or not y_data:
            return
        ax.plot(
            x_data[-1],
            y_data[-1],
            marker='s',
            markersize=MARKER_SIZE + 2,
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=1.5,
            linestyle='none',
        )


    with open(VAR_NUM_NAME, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            if not row or not any(row):  # Skip empty rows
                continue

            var_num_label.append(row[0])
            var_num_series.append(int(row[1]))
            var_num_commit_time.append(float(row[2]))
            var_num_prover_time.append(float(row[3]))
            var_num_proof_size.append(float(row[4]))
            var_num_verifier_time.append(float(row[5]))

    with open(VAR_SIZE_NAME, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            if not row or not any(row):  # Skip empty rows
                continue

            var_size_label.append(row[0])
            var_size_series.append(int(row[1]))
            var_size_commit_time.append(float(row[2]))  # Prover time used as commit time
            var_size_prover_time.append(float(row[3]))  # Same as commit time for var_size
            var_size_proof_size.append(float(row[4]))
            var_size_verifier_time.append(float(row[5]))


    x_tick_depth, x_tick_label_depth = build_ticks(
        [var_num_series],
        [var_num_label],
    )
    x_tick_pars, x_tick_label_pars = build_ticks(
        [var_size_series],
        [var_size_label],
    )

    fig = plt.figure(figsize=(13.5,7.5))
    gs = gridspec.GridSpec(
        3, 4, height_ratios=(1.,1.,-0.6),
        width_ratios=(1.,1.,1.,1.), wspace=0.5, hspace=1.7)



    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    ax5 = plt.subplot(gs[4])
    ax6 = plt.subplot(gs[5])
    ax7 = plt.subplot(gs[6])
    ax8 = plt.subplot(gs[7])

    # ax1.set_aspect('equal', 'box')
    # ax2.set_aspect('equal', 'box')
    # ax3.set_aspect('equal', 'box')
    # ax4.set_aspect('equal', 'box')
    # ax5.set_aspect('equal', 'box')
    # ax6.set_aspect('equal', 'box')

    ax1.set_title('(a) Commit Time \n (log-log, t = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax2.set_title('(b) Open Time \n (log-log, t = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax3.set_title('(c) Proof Size \n (linear-log, t = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax4.set_title('(d) Verifier Time \n (linear-log, t = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)

    ax5.set_title('(e) Commit Time \n (linear-log, N = $2^{30}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax6.set_title('(f) Open Time \n (linear-log, N = $2^{30}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax7.set_title('(g) Proof Size \n (linear-log, N = $2^{30}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax8.set_title('(h) Verifier Time \n (linear-log, N = $2^{30}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)


    ax1.set_xlabel('Total size (N)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax1.set_ylabel('Time (s)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax1.loglog(var_size_series, var_size_commit_time,
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH,
                 label = "BatchPCS",
                 )

    ax1.grid(True, which="major", ls="-", color='0.93')
    ax1.set_xticks(x_tick_pars, x_tick_label_pars)
    ax1.set_ylim(0.1, 5000)

    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax1.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax1.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax1.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax1.spines['left'].set_linewidth(SPLINE_WIDTH)



    ax2.set_xlabel('Total size (N)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax2.set_ylabel('Time (s)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax2.loglog(var_size_series, var_size_prover_time,
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH,
                 label = "ParComm",
                 )

    ax2.grid(True, which="major", ls="-", color='0.93')
    ax2.set_xticks(x_tick_pars, x_tick_label_pars)
    ax2.set_ylim(0.1, 1000)

    ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax2.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax2.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax2.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax2.spines['left'].set_linewidth(SPLINE_WIDTH)


    ax3.set_xlabel('Total size (N)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax3.set_ylabel('Size (KB)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax3.semilogx(var_size_series, var_size_proof_size,
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)

    ax3.grid(True, which="major", ls="-", color='0.93')
    ax3.set_xticks(x_tick_pars, x_tick_label_pars)
    ax3.set_ylim(50, 200)

    ax3.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax3.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax3.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax3.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax3.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax4.set_xlabel('Total size (N)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax4.set_ylabel('Time (ms)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax4.semilogx(var_size_series, var_size_verifier_time,
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax4.grid(True, which="major", ls="-", color='0.93')
    ax4.set_xticks(x_tick_pars, x_tick_label_pars)
    ax4.set_ylim(150, 350)

    ax4.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax4.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax4.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax4.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax4.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax5.set_xlabel('# Vectors (t)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax5.set_ylabel('Time (s)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax5.semilogx(var_num_series, var_num_commit_time,
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax5.grid(True, which="major", ls="-", color='0.93')
    ax5.set_xticks(x_tick_depth, x_tick_label_depth)
    ax5.set_ylim(100,120)

    ax5.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax5.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax5.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax5.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax5.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax6.set_xlabel('# Vectors (t)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax6.set_ylabel('Time (s)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax6.semilogx(var_num_series, var_num_prover_time,
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax6.grid(True, which="major", ls="-", color='0.93')
    ax6.set_xticks(x_tick_depth, x_tick_label_depth)
    ax6.set_ylim(0, 50)

    ax6.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax6.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax6.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax6.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax6.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax7.set_xlabel('# Vectors (t)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax7.set_ylabel('Size (KB)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax7.semilogx(var_num_series, var_num_proof_size,
                 marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax7.grid(True, which="major", ls="-", color='0.93')
    ax7.set_xticks(x_tick_depth, x_tick_label_depth)
    ax7.set_ylim(ax3.get_ylim())

    ax7.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax7.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax7.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax7.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax7.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax8.set_xlabel('# Vectors (t)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax8.set_ylabel('Time (ms)', size = LABEL_SIZE_SINGLE_COLUMN)
    ax8.semilogx(var_num_series, var_num_verifier_time,
                 marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75,
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax8.grid(True, which="major", ls="-", color='0.93')
    ax8.set_xticks(x_tick_depth, x_tick_label_depth)
    ax8.set_ylim(ax4.get_ylim())

    ax8.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax8.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax8.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax8.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax8.spines['left'].set_linewidth(SPLINE_WIDTH)

    # fig.tight_layout()
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=5)



    plt.tight_layout()

    plt.savefig(PIC_NAME,
                transparent=True, bbox_inches='tight', pad_inches=0.02)

    plt.show()


def read_sweep_results(path):
    """Read ordered, verified packed-i8 ParComm sweep results."""
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = [normalize_sweep_row(row) for row in csv.DictReader(handle)]
    rows.sort(key=lambda row: int(row['size_log']))
    if not rows:
        raise ValueError(f'no experiment rows in {path}')
    if any(row['accepted'] != 'true' or row['exit_code'] != '0' for row in rows):
        raise ValueError('result.csv contains an unverified or failed run')
    return rows


def first_value(row, *names):
    """Return the first non-empty value among compatible CSV field names."""
    for name in names:
        value = row.get(name)
        if value not in (None, ''):
            return value
    return ''


def sum_values(row, *names):
    """Sum present numeric fields, or return an empty value when none exist."""
    values = [row.get(name) for name in names]
    present = [value for value in values if value not in (None, '')]
    return '' if not present else f'{sum(map(float, present)):.6f}'


def normalize_sweep_row(row):
    """Normalize legacy packed-i8 and Section 6 result rows for plotting."""
    normalized = dict(row)
    for new_name, legacy_name in (
        ('validity_verify_seconds', 'registration_verify_seconds'),
        ('validity_prove_seconds', 'registration_prove_seconds'),
        ('commit_orchestration_seconds', 'registration_orchestration_seconds'),
    ):
        if not normalized.get(new_name) and normalized.get(legacy_name):
            normalized[new_name] = normalized[legacy_name]
    if not normalized.get('size_log'):
        logical_entries = int(normalized['logical_entries'])
        if logical_entries <= 0 or logical_entries & (logical_entries - 1):
            raise ValueError(
                f'logical_entries={logical_entries} is not a power of two'
            )
        normalized['size_log'] = str(logical_entries.bit_length() - 1)

    normalized['accepted'] = first_value(normalized, 'accepted')
    normalized['exit_code'] = first_value(
        normalized, 'exit_code', 'experiment_exit_code'
    )
    normalized['commit_seconds'] = first_value(
        normalized, 'commit_seconds'
    ) or sum_values(normalized, 'shape_commit_seconds', 'bva_commit_seconds')
    normalized['open_seconds'] = first_value(
        normalized, 'open_seconds', 'opening_prove_seconds'
    ) or sum_values(
        normalized, 'bva_opening_seconds', 'auxiliary_opening_seconds'
    )
    normalized['wall_seconds'] = first_value(
        normalized, 'wall_seconds', 'external_elapsed_seconds', 'total_seconds'
    )
    normalized['verify_seconds'] = first_value(
        normalized, 'verify_seconds', 'total_verifier_seconds'
    ) or sum_values(
        normalized, 'validity_verify_seconds', 'opening_verify_seconds'
    )
    normalized['peak_rss_gib'] = first_value(normalized, 'peak_rss_gib')
    if not normalized['peak_rss_gib']:
        peak_rss_kib = float(
            first_value(normalized, 'peak_rss_kib', 'external_max_rss_kib')
        )
        normalized['peak_rss_gib'] = f'{peak_rss_kib / (1024 ** 2):.6f}'

    for name in (
        'point_commitment_seconds',
        'point_compilation_seconds',
        'packed_eo_reduction_seconds',
        'bva_opening_seconds',
        'final_tbv_seconds',
        'auxiliary_pcs_seconds',
        'auxiliary_opening_seconds',
        'shape_commitment_seconds',
        'dense_vector_commitment_seconds',
        'eo_ip_seconds',
        'eo_ip_accounted_seconds',
        'eo_ip_other_seconds',
        'dense_bva_open_seconds',
        'point_conversion_seconds',
        'tbv_evaluation_seconds',
        'terminal_auxiliary_open_seconds',
        'point_conversion_plus_tbv_evaluation_seconds',
        'commit_orchestration_seconds',
        'opening_orchestration_seconds',
        'other_prover_seconds',
        'prover_breakdown_sum_seconds',
        'pi_ip_proof_size_bytes',
        'pi_a_proof_size_bytes',
    ):
        normalized.setdefault(name, '')
    return normalized


def plot_sweep(rows, output):
    """Plot the ParComm sweep using the Brakefold paper figure format."""
    size_logs = [int(row['size_log']) for row in rows]
    x_values = [1 << size_log for size_log in size_logs]
    x_labels = [rf'$2^{{{size_log}}}$' for size_log in size_logs]
    series = (
        ('Commit Time', 'commit_seconds', 'Time (seconds)', 1.0),
        ('Open Time', 'open_seconds', 'Time (seconds)', 1.0),
        ('End-to-End Time', 'wall_seconds', 'Time (seconds)', 1.0),
        ('Proof Size', 'proof_size_bytes', 'Size (KB)', 1.0 / 1024),
        ('Verifier Time', 'verify_seconds', 'Time (ms)', 1000.0),
        ('Peak RSS', 'peak_rss_gib', 'Memory (GiB)', 1.0),
    )

    fig = plt.figure(figsize=(12.8, 8.6))
    grid = gridspec.GridSpec(
        2,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.32,
        hspace=0.42,
    )
    axes = [plt.subplot(grid[index]) for index in range(6)]
    for index, (axis, (title, field, ylabel, scale)) in enumerate(zip(axes, series)):
        y_values = [float(row[field]) * scale for row in rows]
        axis.loglog(
            x_values,
            y_values,
            base=2,
            color='#0077FF',
            marker='s',
            alpha=0.8,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )
        panel = chr(ord('a') + index)
        axis.set_title(
            f'({panel}) {title} (log-log)',
            size=TITLE_SIZE_SINGLE_COLUMN,
            y=1.0,
        )
        axis.set_xlabel('Total Size (N)', size=LABEL_SIZE_SINGLE_COLUMN, labelpad=4)
        axis.set_ylabel(ylabel, size=LABEL_SIZE_SINGLE_COLUMN)
        axis.set_xticks(x_values, x_labels, size=TICK_SIZE, rotation=25)
        lower_power = 2 ** math.floor(math.log2(min(y_values)))
        upper_power = 2 ** math.ceil(math.log2(max(y_values)))
        axis.set_ylim(lower_power / 1.05, upper_power * 1.05)
        axis.grid(True, which='major', ls='-', color='0.93')
        axis.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        for side in ('top', 'right', 'bottom', 'left'):
            axis.spines[side].set_linewidth(SPLINE_WIDTH)
        axis.set_box_aspect(1)

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.08,
        top=0.96,
        wspace=0.38,
        hspace=0.42,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    """Plot the new sweep by default, with an opt-in legacy figure mode."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=script_dir / 'result.csv')
    parser.add_argument('--output', type=Path, default=script_dir / 'result.png')
    parser.add_argument('--legacy', action='store_true')
    arguments = parser.parse_args()
    if arguments.legacy:
        plot_scatter()
        return
    plot_sweep(read_sweep_results(arguments.input), arguments.output)
    print(f'plotted sweep to {arguments.output}')


if __name__ == '__main__':
    main()
