"""Plot the Fig. 1 in the paper.
"""
import csv
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


PIC_NAME = 'fig1.png'
INPUT_LOOKUP_N_NAME = 'lookup_num_fixed.csv'
INPUT_MUL_N_NAME = 'muls_num_fixed.csv'
INPUT_LOOKUP_NUM_NAME = 'lookup_n_fixed.csv'
INPUT_MUL_NUM_NAME = 'muls_n_fixed.csv'

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
LINE_WIDTH = 0.75

my_cmap = sns.diverging_palette(120., 255., s=100, as_cmap=True)

def plot_scatter():
    """The scatter plot for the RIM results and the closed-form solutions.
    """

    n_label = []
    n_series = []
    num_label = []
    num_series = []

    lookup_n_prover_time = []
    lookup_n_proof_size = []
    lookup_n_verifier_time = []

    mul_n_prover_time = []
    mul_n_proof_size = []
    mul_n_verifier_time = []

    lookup_num_prover_time = []
    lookup_num_proof_size = []
    lookup_num_verifier_time = []

    mul_num_prover_time = []
    mul_num_proof_size = []
    mul_num_verifier_time = []

   
    with open(INPUT_MUL_N_NAME, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:

            n_label.append(row[0])
            n_series.append(int(row[1]))
            mul_n_prover_time.append(float(row[2]))
            mul_n_proof_size.append(float(row[3]))
            mul_n_verifier_time.append(float(row[4])/1000.)

    with open(INPUT_LOOKUP_N_NAME, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:

            lookup_n_prover_time.append(float(row[2]))
            lookup_n_proof_size.append(float(row[3]))
            lookup_n_verifier_time.append(float(row[4])/1000.)

    with open(INPUT_MUL_NUM_NAME, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:

            num_label.append(row[0])
            num_series.append(int(row[1]))
            mul_num_prover_time.append(float(row[2]))
            mul_num_proof_size.append(float(row[3]))
            mul_num_verifier_time.append(float(row[4])/1000.)

    with open(INPUT_LOOKUP_NUM_NAME, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:

            lookup_num_prover_time.append(float(row[2]))
            lookup_num_proof_size.append(float(row[3]))
            lookup_num_verifier_time.append(float(row[4])/1000.)


    x_tick_label = [n_label[i] for i in range(1, len(n_label), 2)]
    x_tick = [n_series[i] for i in range(1, len(n_series), 2)]
    
    fig = plt.figure(figsize=(10,7.5))
    gs = gridspec.GridSpec(
        3, 3, height_ratios=(1.,1.,-0.6),
        width_ratios=(1.,1.,1.), wspace=0.5, hspace=1.7) 



    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    ax5 = plt.subplot(gs[4])
    ax6 = plt.subplot(gs[5])

    # ax1.set_aspect('equal', 'box') 
    # ax2.set_aspect('equal', 'box')
    # ax3.set_aspect('equal', 'box') 
    # ax4.set_aspect('equal', 'box')
    # ax5.set_aspect('equal', 'box') 
    # ax6.set_aspect('equal', 'box')

    ax1.set_title('(a) Prover Time \n (log-log, M = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax2.set_title('(b) Proof Size \n (linear-log, M = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax3.set_title('(c) Verification Time \n (linear-log, M = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax4.set_title('(d) Prover Time \n (log-log, n = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax5.set_title('(e) Proof Size \n (linear-log, n = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)
    ax6.set_title('(f) Verification Time \n (linear-log, n = $2^{10}$)', size = TITLE_SIZE_SINGLE_COLUMN,
                  y = 1.0)


    ax1.set_xlabel('Matrix dimension (n)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax1.set_ylabel('Time (seconds)', size = LABEL_SIZE_SINGLE_COLUMN)  
    ax1.loglog(n_series, mul_n_prover_time, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH,
                 label = "Matrix multiplications",
                 )
    ax1.loglog(n_series, lookup_n_prover_time, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH,
                 label = "Non-linear functions",
                 )
    ax1.grid(True, which="major", ls="-", color='0.93')
    ax1.set_xticks(x_tick, x_tick_label)

    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax1.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax1.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax1.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax1.spines['left'].set_linewidth(SPLINE_WIDTH)
   

    ax2.set_xlabel('Matrix dimension (n)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax2.set_ylabel('Size (KB)', size = LABEL_SIZE_SINGLE_COLUMN)  
    ax2.semilogx(n_series, mul_n_proof_size, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax2.semilogx(n_series, lookup_n_proof_size, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax2.grid(True, which="major", ls="-", color='0.93')
    ax2.set_xticks(x_tick, x_tick_label, size = TICK_SIZE)
    ax2.set_ylim(0, 850)

    ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax2.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax2.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax2.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax2.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax3.set_xlabel('# Matrix dimension (n)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax3.set_ylabel('Time (seconds)', size = LABEL_SIZE_SINGLE_COLUMN)  
    ax3.semilogx(n_series, mul_n_verifier_time, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax3.semilogx(n_series, lookup_n_verifier_time, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax3.grid(True, which="major", ls="-", color='0.93')
    ax3.set_xticks(x_tick, x_tick_label, size = TICK_SIZE)
    ax3.set_ylim(0, 2.3)

    ax3.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax3.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax3.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax3.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax3.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax4.set_xlabel('# Matrices (M)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax4.set_ylabel('Time (seconds)', size = LABEL_SIZE_SINGLE_COLUMN)  
    ax4.loglog(num_series, mul_num_prover_time, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax4.loglog(num_series, lookup_num_prover_time, 
               marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax4.grid(True, which="major", ls="-", color='0.93')
    ax4.set_xticks(x_tick, x_tick_label, size = TICK_SIZE)
    ax4.set_ylim(ax1.get_ylim())
    
    ax4.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax4.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax4.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax4.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax4.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax5.set_xlabel('# Matrices (M)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax5.set_ylabel('Size (KB)', size = LABEL_SIZE_SINGLE_COLUMN)  
    ax5.semilogx(num_series, mul_num_proof_size, 
                 marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax5.semilogx(num_series, lookup_num_proof_size, 
                 marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax5.grid(True, which="major", ls="-", color='0.93')
    ax5.set_xticks(x_tick, x_tick_label, size = TICK_SIZE)
    ax5.set_ylim(ax2.get_ylim())

    ax5.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax5.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax5.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax5.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax5.spines['left'].set_linewidth(SPLINE_WIDTH)

    ax6.set_xlabel('# Matrices (M)', size = LABEL_SIZE_SINGLE_COLUMN,
                   labelpad=4)
    ax6.set_ylabel('Time (seconds)', size = LABEL_SIZE_SINGLE_COLUMN)  
    ax6.semilogx(num_series, mul_num_verifier_time, 
                 marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax6.semilogx(num_series, lookup_num_verifier_time, 
                 marker = MARKER_STYLE, markersize = MARKER_SIZE,
                 color = COLOR2, alpha = 0.75, 
                 linestyle = LINE_STYLE, linewidth = LINE_WIDTH)
    ax6.grid(True, which="major", ls="-", color='0.93')
    ax6.set_xticks(x_tick, x_tick_label, size = TICK_SIZE)
    ax6.set_ylim(ax3.get_ylim())

    ax6.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax6.spines['top'].set_linewidth(SPLINE_WIDTH)
    ax6.spines['right'].set_linewidth(SPLINE_WIDTH)
    ax6.spines['bottom'].set_linewidth(SPLINE_WIDTH)
    ax6.spines['left'].set_linewidth(SPLINE_WIDTH)
    # fig.tight_layout()
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=5)



    # Add a single legend for all subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=3, fontsize=LEGEND_FONT_SIZE, frameon=False,  bbox_to_anchor=(0.5, 0.03))
    plt.tight_layout()

    plt.savefig(PIC_NAME, 
                transparent=True, bbox_inches='tight', pad_inches=0.02)

    plt.show()


if __name__ == '__main__':

    plot_scatter()
