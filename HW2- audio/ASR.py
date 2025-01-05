import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Check if the sampling rate is 16kHz
def checkfs(filepath):
    _, fs = librosa.load(filepath, sr=None)
    if fs != 16000:
        raise ValueError('Sampling rate is not 16kHz')

# Calculate the Mel spectrogram of the audio file
def calc_mel_spectrogram(filepath):
    y, fs = librosa.load(filepath, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=fs , hop_length=10*fs//1000, win_length=25*fs//1000, n_mels=80)
    return S

# Get the mel spectrogram of all the files in the directory
def get_mels_dir(dirpath):
    set_mel = {}
    files = os.listdir(dirpath)
    for file in files:
        filepath = os.path.join(dirpath, file)
        checkfs(filepath)
        S = calc_mel_spectrogram(filepath)
        set_mel[file] = S
    return set_mel


# Save the Mel spectrogram of the audio file
def save_mel(mel , filename):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel spectrogram of {filename}')
    plt.tight_layout()
    plt.savefig(f'mel_outputs_Q2d/{filename[0:-4]}.png')
    plt.close()

# Save the Mel spectrogram of all the files in the directory
def save_mels_dir(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        mel = calc_mel_spectrogram(filepath)
        save_mel(mel, filename)

 # Calculate the DTW distance between two Mel spectrograms
import numpy as np

def dtw(mel_data, mel_representative):
    # Step 1: Create the local distance matrix
    distances_matrix = np.zeros((mel_representative.shape[1], mel_data.shape[1]))
    for i in range(distances_matrix.shape[0]):
        for j in range(distances_matrix.shape[1]):
            distances_matrix[i, j] = np.linalg.norm(
                mel_representative[:, i] - mel_data[:, j]
            )
    
    # Step 2: Create the DTW matrix and the back-pointer matrix
    dtw_matrix = np.zeros_like(distances_matrix)
    backtrace = np.zeros_like(distances_matrix, dtype=np.int8)
    # backtrace coding: 0 => (i-1,j), 1 => (i,j-1), 2 => (i-1,j-1)

    # Initialization
    dtw_matrix[0, 0] = distances_matrix[0, 0]

    # Fill first row
    for j in range(1, dtw_matrix.shape[1]):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + distances_matrix[0, j]
        backtrace[0, j] = 1  # from (0, j-1)

    # Fill first column
    for i in range(1, dtw_matrix.shape[0]):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + distances_matrix[i, 0]
        backtrace[i, 0] = 0  # from (i-1, 0)

    # Step 3: Fill the rest of the matrix
    for i in range(1, dtw_matrix.shape[0]):
        for j in range(1, dtw_matrix.shape[1]):
            cost_up = dtw_matrix[i-1, j]
            cost_left = dtw_matrix[i, j-1]
            cost_diag = dtw_matrix[i-1, j-1]
            mincost = min(cost_up, cost_left, cost_diag)

            dtw_matrix[i, j] = distances_matrix[i, j] + mincost
            
            if mincost == cost_up:
                backtrace[i, j] = 0  # from (i-1, j)
            elif mincost == cost_left:
                backtrace[i, j] = 1  # from (i, j-1)
            else:
                backtrace[i, j] = 2  # from (i-1, j-1)

    # Step 4: Backtrack to find the full alignment path
    i = dtw_matrix.shape[0] - 1
    j = dtw_matrix.shape[1] - 1
    path = [(i, j)]

    while not (i == 0 and j == 0):
        move = backtrace[i, j]
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
        path.append((i, j))

    path.reverse()

    return dtw_matrix[-1, -1], path



def collapse_function_B(seq):
    collapse_seq = []
    last_char = ""
    for c in seq:
        if c != '^':
            if c != last_char:
                collapse_seq.append(c)
        last_char = c
    return "".join(collapse_seq)


def forward_pass(pred, label_mapping, GT, summize=True):
    op_map = {val: key for key, val in label_mapping.items()}
    alpha_matrix = np.zeros(shape=(len(GT),pred.shape[0]), dtype=np.float32)
    backtrace_matrix = np.zeros(alpha_matrix.shape, dtype=np.int32)
    alpha_matrix[0][0] = pred[0][op_map[GT[0]]]
    alpha_matrix[1][0] = pred[0][op_map[GT[1]]]
    
    rows , cols = alpha_matrix.shape
    for col in range (1, cols):
        min_row = 0
        max_row = rows
        if col <= cols//2:
            max_row = min(rows, (col+1)*2)
        if col >= cols // 2:
            min_row = max(0, rows - 2 * (cols - col) )
        
        for row in range(min_row , max_row):
            label = op_map[GT[row]]
            prob = pred[col][label]
            for i in range(max(0, row - 2), row + 1):
                if (i == row -2 and GT[row] == '^'):
                    continue
                if(summize):
                    alpha_matrix[row][col] += prob * alpha_matrix[i][col - 1]
                else:
                    if alpha_matrix[row][col] <= prob * alpha_matrix[i][col - 1]:
                        alpha_matrix[row][col] = prob * alpha_matrix[i][col - 1]
                        backtrace_matrix[row][col] = i

    if not summize:
        best_path = []   
        if alpha_matrix[-1, -1] < alpha_matrix[-2 , -1]:
            final_index = rows - 2
        else:
            final_index = rows - 1
        curr_cell = final_index
        for col in range(cols - 1 , -1 , -1):
            best_path.append(curr_cell)
            curr_cell = backtrace_matrix[curr_cell][col]
        
        return alpha_matrix[final_index, -1] , alpha_matrix , backtrace_matrix , best_path[::-1]

    return alpha_matrix[-1, -1] + alpha_matrix[-2 , -1]  , alpha_matrix 
        
        
    
def plot_matrix(matrix ,row_labels,col_labels,save_path):
    
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='plasma') 
    plt.colorbar(cax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    plt.xticks(rotation=0, ha='left')

    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color='white')
     # If a path is provided, save the figure
    if save_path is not None:
        plt.savefig(save_path, dpi=150)  # you can adjust dpi for higher resolution
        plt.close(fig)
    else:
        # If no path is specified, just display it on screen
        plt.show()
    


def plot_text_matrix_with_coloring(
    text_matrix, 
    color_array, 
    highlight_dict=None,  # {col_index: row_index} to highlight in red
    row_labels=None, 
    col_labels=None, 
    cmap='viridis'
):
    """
    Plots a text matrix and color-codes each column based on the values in color_array.
    Additionally, highlights one cell in each column (as specified by highlight_dict) in red.
    """
    rows, cols = text_matrix.shape
    
    # 1) Validate color_array length
    if len(color_array) != cols:
        raise ValueError(
            f"color_array length ({len(color_array)}) must match "
            f"the number of columns ({cols})."
        )

    # 2) Provide default labels if none are given
    if row_labels is None:
        row_labels = [f"r{i}" for i in range(rows)]
    if col_labels is None:
        col_labels = [f"c{j}" for j in range(cols)]
    
    # 3) Create a 2D numeric matrix to color columns
    color_matrix = np.tile(color_array, (rows, 1))

    # 4) Plot
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
    im = ax.imshow(color_matrix, cmap=cmap, aspect='auto', origin='upper')

    # 5) Put text in each cell
    for i in range(rows):
        for j in range(cols):
            ax.text(
                j, i, text_matrix[i, j],
                ha='center', va='center',
                color='white', fontsize=10
            )

    # 6) Highlight specific cells in red if highlight_dict is given
    if highlight_dict:
        for col_idx, row_idx in highlight_dict.items():
            # Draw a red rectangle over the cell
            rect = Rectangle(
                (col_idx - 0.5, row_idx - 0.5),  # lower-left corner
                1, 1,                           # width, height
                fill=True, 
                facecolor='red', 
                alpha=0.4,                      # slight transparency so text is visible
                edgecolor='red',
                linewidth=2,
                zorder=3                        # draw on top
            )
            ax.add_patch(rect)

    # 7) Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Column Color Intensity')

    # 8) Ticks & labels
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()








def main():
    # Directory path
    dirpath_Training = 'audio_files/Training_Set'
    dirpath_Testing = 'audio_files/Evaluation_Set'
    dirpath_Representative = 'audio_files/Representative'

    #Question 2(d):Save all the Mel spectrograms

    #saveall_mel(dirpath_Training)
    #saveall_mel(dirpath_Testing)
    #saveall_mel(dirpath_Representative)

    #question 3 : DTW

    #Training_database = get_mels_dir(dirpath_Training)
    #Testing_database = get_mels_dir(dirpath_Testing)
    #reference_database = get_mels_dir(dirpath_Representative)

    #Question 4 : CTC collapse functiob B
    #def collapse_function_B(seq)

    #Question 5 (a)(b) : forward pass of the CTC algorithm
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00
    label_mapping = {0: 'a', 1:'b', 2:'^'}
    GT = "aba"
    GT_blanks = '^' + '^'.join(list(GT)) + '^'
    # Question 5 (c) :
    prob , alpha_matrix = forward_pass(pred , label_mapping ,GT_blanks)
    print("Q5 (c): ")
    print(f"The probability of the sequence aba is {prob}")

    # Question 5 (d) plot the pred matrix:
    row_labels = list(GT_blanks)
    col_lables = [i for i in range(0 , pred.shape[0])]
    plot_matrix(alpha_matrix , row_labels , col_lables ,"Forward_Algorithm_plots/Q5d_pred_matrix.png" )

    # Question 6 : forward pass of the CTC for force alignment
    prob, alpha_matrix, backtrace_matrix , best_path = forward_pass(pred, label_mapping, GT_blanks, False)

    #Question 6 (b): The most probable path
    print("Q6 (b): ")
    path_lables = [GT_blanks[i] for i in best_path]
    print(f"The most probable path is : {best_path} , {path_lables}")
    
    # Question 6 (c): probability of best path
    print("Q6 (c): ")
    print(f"the probability of the best path is: {prob}")

    # Question 6 (d): plot the aligned sequence
    plot_matrix(alpha_matrix , row_labels , path_lables , "Forward_Algorithm_plots/Q6d_aligned_sequence.png")

    # Question 6 (e): plot the backtrace matrix, and the selected path.
    char_matrix = np.empty_like(backtrace_matrix, dtype=object)
    for i in range(backtrace_matrix.shape[0]):
        for j in range(backtrace_matrix.shape[1]):
            if (i == 0 and j == 0 or j == 0 and i == 1):
                char_matrix[i, j] = "START"
            else:  
                x = backtrace_matrix[i, j]  
                if(x == 0):
                    char_matrix[i, j] = "N/A"
                else:
                    char_matrix[i, j] = GT_blanks[x]  

    # Plot:
    plot_text_matrix_with_coloring(
    char_matrix, 
    best_path, 
    row_labels=row_labels, 
    col_labels=col_lables,
    cmap='plasma'
)
    #plot_matrix(char_matrix , row_labels , col_lables , "Forward_Algorithm_plots/Q6e_backtrace_matrix_path.png")

    #Question 7 : force_align.pkl
    data = pkl.load(open('$PATH', 'rb'))



    
    



# Run your main
if __name__ == "__main__":
    main()
    print("Done")
 