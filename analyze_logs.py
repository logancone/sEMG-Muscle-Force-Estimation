from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import math
import pandas as pd
from statsmodels.stats.anova import AnovaRM


#Save each entry as [model, test_subj, val_subj, pre_tl_loss, post_tl_loss, num_epochs, path]
def compile_logs():
    for entry in Path('logs/').iterdir():
        if str(entry)[26:] == 'CNN':
            model = 1
        elif str(entry)[26:] == 'CLSTM':
            model = 2
        elif str(entry)[26:] == 'TCN':
            model = 3
        else:
            continue
    # for entry in Path('new_logs/').iterdir():
    #     if str(entry)[30:] == 'CNN':
    #         model = 1
    #     elif str(entry)[30:] == 'CLSTM':
    #         model = 2
    #     elif str(entry)[30:] == 'TCN':
    #         model = 3
    #     else:
    #         continue

        raw_log = Path.read_text(Path(f"{entry}/logger.log"))
        test_subj = int(raw_log.splitlines()[1][17:19])
        val_subj = int(raw_log.splitlines()[1][37:41])

        loss_id = raw_log.find("Test | Loss: ")
        pre_tl_loss = math.sqrt(float(raw_log[loss_id+13:loss_id+20]))
        post_tl_loss = math.sqrt(float(raw_log[loss_id+44:loss_id+51]))

        epoch_id = raw_log.find("Early stopping triggered on epoch ")
        if epoch_id > 0:
            num_epochs = int(raw_log[epoch_id+34: epoch_id+36])
        else:
            num_epochs = 100

        # print((model, test_subj, val_subj, pre_tl_loss, post_tl_loss, num_epochs, entry))
        runs.append((model, test_subj, val_subj, pre_tl_loss, post_tl_loss, num_epochs, entry))

def print_avail_subj_combs():
    unused_combs = []
    for i in range(17):
        for j in range(17):
            if j != i:
                unused_combs.append((i+1, j+1))

    for run in runs:
        if (run[1], run[2]) in unused_combs:
            unused_combs.remove((run[1], run[2]))

    for comb in unused_combs:
        print(comb)

def get_avg_loss(exclude_outlier: bool = True):
    cnn_no_tl = []
    cnn_tl = []
    clstm_no_tl = []
    clstm_tl = []
    tcn_no_tl = []
    tcn_tl = []

    for run in runs:
        if exclude_outlier == True:
            if run[1] == 9 or run[1] == 5:
                continue

        if run[0] == 1:
            cnn_no_tl.append(run[3])
            cnn_tl.append(run[4])
        elif run[0] == 2:
            clstm_no_tl.append(run[3])
            clstm_tl.append(run[4])
        elif run[0] == 3:
            tcn_no_tl.append(run[3])
            tcn_tl.append(run[4])

    cnn_pre = sum(cnn_no_tl) / len(cnn_no_tl)
    cnn_post = sum(cnn_tl) / len(cnn_tl)

    clstm_pre = sum(clstm_no_tl) / len(clstm_no_tl)
    clstm_post = sum(clstm_tl) / len(clstm_tl)

    tcn_pre = sum(tcn_no_tl) / len(tcn_no_tl)
    tcn_post = sum(tcn_tl) / len(tcn_tl)

    # Print results
    print(f"CNN: Pre-TL: {cnn_pre} | Post-TL: {cnn_post}")
    print(f"C-LSTM: Pre-TL: {clstm_pre} | Post-TL: {clstm_post}")
    print(f"TCN: Pre-TL: {tcn_pre} | Post-TL: {tcn_post}")

    # Return as a tuple
    return (cnn_pre, cnn_post, clstm_pre, clstm_post, tcn_pre, tcn_post)

def get_avg_loss_error(exclude_outlier: bool = True):
    cnn_no_tl = []
    cnn_tl = []
    clstm_no_tl = []
    clstm_tl = []
    tcn_no_tl = []
    tcn_tl = []

    for run in runs:
        if exclude_outlier == True:
            if run[1] == 9 or run[1] == 5:
                continue

        if run[0] == 1:
            cnn_no_tl.append(run[3])
            cnn_tl.append(run[4])
        elif run[0] == 2:
            clstm_no_tl.append(run[3])
            clstm_tl.append(run[4])
        elif run[0] == 3:
            tcn_no_tl.append(run[3])
            tcn_tl.append(run[4])
    
    big_list = []
    avg_list = []
    error_list = []

    # Return as a tuple
    big_list.extend([np.array(cnn_no_tl), np.array(cnn_tl), np.array(clstm_no_tl), np.array(clstm_tl), np.array(tcn_no_tl), np.array(tcn_tl)])
    for i in big_list:
        avg_list.append(sum(i)/len(i))
        error_list.append(np.std(i, ddof=1) / np.sqrt(len(i)))
    
    return (avg_list, error_list) 

def find_dupe():
    combos = []
    for run in runs:
        if (run[0], run[1], run[2]) not in combos:
            combos.append((run[0], run[1], run[2]))
        else:
            print(f"Dupe at: {run[6]}")

def find_loss_under_x(target_loss: float):
    included_runs = []
    for run in runs:
        if run[4] <= target_loss or run[3] <= target_loss:
            included_runs.append(run)

    for run in included_runs:
        print(run)

def find_incomplete_combs():
    attempted_cnn = []
    attempted_clstm = []
    attempted_tcn = []

    attempted_all = []
    uneven = []
    for run in runs:
        if run[0] == 1:
            attempted_cnn.append((run[1], run[2]))
        elif run[0] == 2:
            attempted_clstm.append((run[1], run[2]))
        elif run[0] == 3:
            attempted_tcn.append((run[1], run[2]))
    
    for run in runs:
        if (run[1], run[2]) in attempted_cnn and (run[1], run[2]) in attempted_clstm and (run[1], run[2]) in attempted_tcn:
            if (run[1], run[2]) not in attempted_all:
                attempted_all.append((run[1], run[2]))
        else:
            uneven.append(run)

    # print(uneven)
    for r in uneven:
        print(r)

    print(len(attempted_all))

def get_runs_for_comb(test_id, val_id):
    desired_runs = []
    for run in runs:
        if run[1] == test_id and run[2] == val_id:
            desired_runs.append(run)

    prev_model = 0
    for r in desired_runs:
        # print(r)
        assert r[0] > prev_model

    # print(desired_runs)
    return desired_runs

def get_loss_by_test_pretl(test_id):
    cnn = []
    clstm = []
    tcn = []

    test_nums = 0

    for i in range(17):
        t = get_runs_for_comb(test_id, i+1)
        if len(t) == 3:
            cnn.append(t[0][3])
            clstm.append(t[1][3])
            tcn.append(t[2][3])

            test_nums += 1
        elif len(t) > 0:
            print("BAD")

    return (sum(cnn)/len(cnn), sum(clstm)/len(clstm), sum(tcn)/len(tcn), test_nums)

def get_loss_by_test_posttl(test_id):
    cnn = []
    clstm = []
    tcn = []

    test_nums = 0

    for i in range(17):
        t = get_runs_for_comb(test_id, i+1)
        if len(t) == 3:
            cnn.append(t[0][4])
            clstm.append(t[1][4])
            tcn.append(t[2][4])

            test_nums += 1
        elif len(t) > 0:
            print("BAD")

    return (sum(cnn)/len(cnn), sum(clstm)/len(clstm), sum(tcn)/len(tcn), test_nums)

def graph_loss_by_test():
    
    ids = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17")
    cnn_loss = []
    clstm_loss = []
    tcn_loss = []
    for i in range(17):
        loss = get_loss_by_test_posttl(i+1)

        cnn_loss.append(loss[0])
        clstm_loss.append(loss[1])
        tcn_loss.append(loss[2])
    

    model_losses = {
        'CNN': cnn_loss,
        'C-LSTM': clstm_loss,
        'TCN': tcn_loss,
    }

    x = np.arange(len(ids))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    i = 1
    for attribute, measurement in model_losses.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Loss by subject')
    ax.set_xticks(x + width, ids)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, .5)

    plt.show()

def get_loss_by_val(val_id):
    cnn = []
    clstm = []
    tcn = []

    test_nums = 0

    for i in range(17):
        t = get_runs_for_comb(i+1, val_id)
        if len(t) == 3:
            cnn.append(t[0][4])
            clstm.append(t[1][4])
            tcn.append(t[2][4])

            test_nums += 1
        elif len(t) > 0:
            print("BAD")

    return (sum(cnn)/len(cnn), sum(clstm)/len(clstm), sum(tcn)/len(tcn), test_nums)

def graph_loss_by_val():
    
    ids = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17")
    cnn_loss = []
    clstm_loss = []
    tcn_loss = []
    for i in range(17):
        loss = get_loss_by_val(i+1)

        cnn_loss.append(loss[0])
        clstm_loss.append(loss[1])
        tcn_loss.append(loss[2])
    

    model_losses = {
        'CNN': cnn_loss,
        'C-LSTM': clstm_loss,
        'TCN': tcn_loss,
    }

    x = np.arange(len(ids))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    i = 1
    for attribute, measurement in model_losses.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Loss by val subject')
    ax.set_xticks(x + width, ids)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, .5)

    plt.show()

def run_stats_tests(post_tl: bool = True):
    subj_list = []
    model_list = []
    rmse_list = []

    for i in range(1, 18):
        if i == 5 or i == 9:
            continue
        subj_list.extend([i, i, i])
        model_list.extend(['CNN', 'CLSTM', 'TCN'])
        if post_tl:
            loss = get_loss_by_test_posttl(i)
        else:
            loss = get_loss_by_test_pretl(i)
        rmse_list.extend([loss[0], loss[1], loss[2]])

    data = pd.DataFrame({
    'subject' : subj_list,
    'model' : model_list,
    'rmse' : rmse_list
    })

    anova = AnovaRM(data, 'rmse', 'subject', within=['model'])
    result = anova.fit()

    if post_tl:
        print("After TL")
    else:
        print("Before TL")
    print(result.anova_table)

def find_subj_outliers():
    subj_loss = []
    for i in range(17):
        loss_pre = get_loss_by_test_pretl(i+1)
        loss_post = get_loss_by_test_posttl(i+1)
        t = (loss_pre[0] + loss_pre[1] + loss_pre[2] + loss_post[0] + loss_post[1] + loss_post[2]) / 6
        subj_loss.append(t)

    rmse_values = np.array(subj_loss)
    print(rmse_values)

    Q1 = np.percentile(rmse_values, 25)
    Q3 = np.percentile(rmse_values, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = np.where((rmse_values < lower_bound) | (rmse_values > upper_bound)) 
    print("Outliers:", outliers)

def graph_loss_by_model():
    # Font properties
    font_regular = fm.FontProperties(fname=r"C:\Users\logan\AppData\Local\Microsoft\Windows\Fonts\QuattrocentoSans-Regular.ttf")
    font_bold = fm.FontProperties(fname=r"C:\Users\logan\AppData\Local\Microsoft\Windows\Fonts\QuattrocentoSans-Bold.ttf")

    # EXCLUDES OUTLIERS!!!
    models = ("CNN", "C-LSTM", "TCN")    

    # loss = get_avg_loss(True)

    # model_losses = {
        # 'Before TL': [loss[0], loss[2], loss[4]],
    #     'After TL': [loss[1], loss[3], loss[5]]
    # }

    loss = get_avg_loss_error(True)
    model_losses = {
        'Before TL': [loss[0][0], loss[0][2], loss[0][4]],
        'After TL': [loss[0][1], loss[0][3], loss[0][5]]
    }

    errors = [
        [loss[1][0], loss[1][2], loss[1][4]],
        [loss[1][1], loss[1][3], loss[1][5]]
    ]

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    colors = ['#664F93', '#3684A0']

    fig, ax = plt.subplots(layout='constrained')

    multiplier = 0
    for attribute, measurement in model_losses.items():
        
        offset = width * multiplier
        
        rects = ax.bar(
            x + offset, measurement, width,
            label=attribute,
            color=colors[multiplier],
            edgecolor='black',
            alpha=0.9,
            yerr=errors[multiplier],
            capsize=4
        )
        # ax.bar_label(rects, padding=3, fmt='%.4f', fontsize=8, fontproperties=font_regular)
        multiplier += 1
  
    ax.set_ylabel('Average Loss (RMSE)', fontproperties=font_bold)
    ax.set_xlabel('Model Architecture', fontproperties=font_bold)
    ax.set_title('Average RMSE by Model', fontproperties=font_bold, fontsize=16)
    ax.set_xticks(x + width/2, models, fontproperties=font_bold)
    ax.set_ylim(.1, .125)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, prop=font_regular)
    ax.legend(loc='best', ncol=2, frameon=False, prop=font_bold)
    

    plt.show()

def make_diversity_chart():
    import matplotlib.pyplot as plt

    counts = [6, 4, 5, 3]  # replace with your actual data
    labels = ['Male & Lifts', 'Male & Doesn’t Lift', 'Female & Lifts', 'Female & Doesn’t Lift']
    colors = ['#3684A0', "#306F85", '#664F93', "#5E4B86"]  # Blue = Male, Purple = Female

    # Optional: slightly darker shade for "doesn’t lift"
    # explode = (0.05, 0, 0.05, 0)  # separate “lifts” slices a bit for visual interest

    fig, ax = plt.subplots()
    ax.pie(
        counts,
        labels=labels,
        autopct='%1.0f%%',
        startangle=90,
        colors=colors,
        # explode=explode,
        wedgeprops={'edgecolor': 'black'}
    )
    ax.set_title('Participant Demographics: Sex × Weightlifting', fontsize=14, fontweight='bold')

    plt.show()


def plot_normalized_data_subj(subj_id):
    # Font properties
    font_regular = fm.FontProperties(fname=r"C:\Users\logan\AppData\Local\Microsoft\Windows\Fonts\QuattrocentoSans-Regular.ttf")
    font_bold = fm.FontProperties(fname=r"C:\Users\logan\AppData\Local\Microsoft\Windows\Fonts\QuattrocentoSans-Bold.ttf")

    raw_data = np.load(f"raw_data/raw_data_compiled/Subject_{subj_id}_Compiled.npz")
    header = raw_data['header']
    semgVals = raw_data['semgVals']
    forceVals = raw_data['forceVals']
    forceIdxs = raw_data['forceIdxs']

    max_semg = header[1]
    max_force = header[2]
    
    # ReLU the forceVals (set negatives to 0 since negative force just means force plate is lifting off of backplate meaning 0 force)
    for i in range(forceVals.size):
        if forceVals[i] < 0:
            forceVals[i] = 0

    # Remove any semg values after the final force timestamp (may be up to 90 values, useless since cannot interpolate without low and high point)
    maxIdx = forceIdxs[forceIdxs.size - 1]
    if semgVals.size > maxIdx + 1:
        semgVals = semgVals[:maxIdx+1]

    # Min-max normalization function
    def normalize_value(num, min_value, max_value):
        if max_value - min_value == 0:
            print("No range!!")
            new_num = 0
        else:
            new_num = (num - min_value) / (max_value - min_value)
        return new_num

    normalized_semg_temp = []
    normalized_force_temp = []

    # Normalize semg data with subjects max semg value:
    for i in range(semgVals.size):
        normalized_semg_temp.append(normalize_value(semgVals[i], 0, max_semg))

    # Normalize force data with subject max force value
    for i in range(forceVals.size):
        normalized_force_temp.append(normalize_value(forceVals[i], 0, max_force))

    # Set normalized vals to their own array
    normalized_semg = np.array(normalized_semg_temp)
    normalized_force = np.array(normalized_force_temp)

    # left_bound = 20000
    # right_bound = 30000
    left_bound = 87000
    right_bound = 100000
    semg = normalized_semg[left_bound:right_bound]
    force = []
    f_id = []

    for i in range(len(normalized_force)):
        if forceIdxs[i] >= left_bound and forceIdxs[i] <= right_bound:
            f_id.append(forceIdxs[i] - left_bound)
            force.append(normalized_force[i])


    fig, ax = plt.subplots(layout='constrained', figsize=(6,3))

    ax.plot(semg, label='sEMG', color='#3684A0')
    ax.plot(f_id, force, label='Force', color='#E64B3C')

    ax.set_ylabel('Normalized Value (0-1)', fontproperties=font_bold)
    ax.set_xlabel('Timestamp (ms)', fontproperties=font_bold)
    ax.set_title('Outlier Sample: Incomplete sEMG', fontproperties=font_bold, fontsize=16)
    # ax.set_xticks(x + width/2, models, fontproperties=font_regular)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', ncol=1, frameon=False, prop=font_bold, fontsize=8)


    plt.show()

def count_male_and_female():
    male = 0
    female = 0
    for i in range(17):
        data = np.load(f'processed_data/Subject_{i+1}_Processed.npz')
        sex = data['header'][4]
        if sex == 0:
            male += 1
        elif sex == 1:
            female += 1
        else:
            print("BAD")

    print(f"Male: {male} | Female: {female}")

def get_ages():
    for i in range(17):
        data = np.load(f'processed_data/Subject_{i+1}_Processed.npz')
        age = data['header'][3]
        print(age)

def get_lw():
    for i in range(17):
        data = np.load(f'processed_data/Subject_{i+1}_Processed.npz')
        lw = bool(data['header'][7])
        print(data['header'])

if __name__ == '__main__':
    runs = []
    compile_logs()
    # print_avail_subj_combs()
    # find_dupe()
    # get_avg_loss()
    # find_loss_under_x(.09)
    # find_incomplete_combs()
    # get_runs_for_comb(1, 2)
    # graph_loss_by_test()
    # run_stats_tests(False)
    # graph_loss_by_val()
    # find_subj_outliers()
    # graph_loss_by_model()
    # make_diversity_chart()
    # for i in range(17):
        # plot_normalized_data_subj(i+1)
        # get_runs_for_comb(10, i+1)

    # plot_normalized_data_subj(7)
    # plot_normalized_data_subj(8)
    # plot_normalized_data_subj(10)
    # plot_normalized_data_subj(14)
    # plot_normalized_data_subj(15)

    # plot_normalized_data_subj(9)

    # count_male_and_female()

    # get_ages()
    get_lw()
