import librosa
import numpy as np
import os
import matplotlib.pyplot as plt


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
def dtw(mel_data, mel_representative):
    path = []
    distances_matrix = np.zeros((mel_representative.shape[1], mel_data.shape[1]))

    for i in range(distances_matrix.shape[0]):
        for j in range(distances_matrix.shape[1]):
            distances_matrix[i, j] = np.linalg.norm(mel_representative[:, i] - mel_data[:, j])
    print(distances_matrix)
    dtw_matrix = np.zeros(distances_matrix.shape)
    dtw_matrix[0, 0] = distances_matrix[0, 0]
    for i in range(0 , dtw_matrix.shape[0]):
        for j in range(0 , dtw_matrix.shape[1]):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                dtw_matrix[i, j] = distances_matrix[i, j] + dtw_matrix[i, j-1]
                path.append((i, j-1))
            elif j == 0:
                dtw_matrix[i, j] = distances_matrix[i, j] + dtw_matrix[i-1, j]
                path.append((i-1, j))
            else:
                mincost = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
                dtw_matrix[i, j] = distances_matrix[i, j] + mincost
                if dtw_matrix[i-1, j] == mincost:
                    path.append((i-1, j))
                elif dtw_matrix[i, j-1] == mincost:
                    path.append((i, j-1))
                else:
                    path.append((i-1, j-1))
    return dtw_matrix[-1,-1], path


def create_training_dist_matrix(train_db, ref_db):
    dist_matrix = np.zeros((4, 10, 10))
    names = set([key[:-6] for key in train_db.keys()])
    ref = list(ref_db.keys())[0][:-6]
    for i, name in enumerate(names):
        for j in range(10):
            dist_matrix[i, j, :] = [dtw(train_db[f"{name}_{j}.wav"], ref_db[f"{ref}_{k}.wav"])[0] for k in range(10)]
    return dist_matrix


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
        
        
    
def plot_matrix(matrix ,row_labels,col_labels):
    
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
    print(f"The probability of the sequence aba is {prob}")

    # Question 5 (d) plot the pred matrix:
    row_labels = list(GT_blanks)
    col_lables = [i for i in range(0 , pred.shape[0])]
    plot_matrix(alpha_matrix , row_labels , col_lables)

    # Question 6 : forward pass of the CTC for force alignment
    prob, alpha_matrix, backtrace_matrix , best_path = forward_pass(pred, label_mapping, GT_blanks, False)

    #Question 6 (b): The most probable path
    print(f"The most probable path is : {best_path}")
    
    # Question 6 (c): probability of best path
    print(f"the probability of the best path is: {prob}")

    # Question 6 (d): plot the aligned sequence
    col_lables = [GT_blanks[i] for i in best_path]
    plot_matrix(alpha_matrix , row_labels , col_lables)

    # Question 6 (e): plot the backtrace matrix, and the selected path.
    col_lables = [i for i in range(0 , pred.shape[0])]
    plot_matrix(backtrace_matrix , row_labels , col_lables)


# Run your main
if __name__ == "__main__":
    # main()
    dirpath_Training = 'audio_files/Training_Set'
    dirpath_Testing = 'audio_files/Evaluation_Set'
    dirpath_Representative = 'audio_files/Representative'
    Training_database = get_mels_dir(dirpath_Training)
    Testing_database = get_mels_dir(dirpath_Testing)
    reference_database = get_mels_dir(dirpath_Representative)
    dist_matrix = create_training_dist_matrix(Training_database, reference_database)
    print(dist_matrix)
    print("Done")
 