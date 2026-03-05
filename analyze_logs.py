from pathlib import Path

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

        raw_log = Path.read_text(Path(f"{entry}/logger.log"))
        test_subj = int(raw_log.splitlines()[1][17:19])
        val_subj = int(raw_log.splitlines()[1][37:41])

        loss_id = raw_log.find("Test | Loss: ")
        pre_tl_loss = float(raw_log[loss_id+13:loss_id+20])
        post_tl_loss = float(raw_log[loss_id+44:loss_id+51])

        epoch_id = raw_log.find("Early stopping triggered on epoch ")
        num_epochs = int(raw_log[epoch_id+34: epoch_id+36])

        # print((model, test_subj, val_subj, pre_tl_loss, post_tl_loss, num_epochs))
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

def print_avg_loss():
    cnn_no_tl = []
    cnn_tl = []
    clstm_no_tl = []
    clstm_tl = []
    tcn_no_tl = []
    tcn_tl = []

    for run in runs:
        if run[0] == 1:
            cnn_no_tl.append(run[3])
            cnn_tl.append(run[4])
        elif run[0] == 2:
            clstm_no_tl.append(run[3])
            clstm_tl.append(run[4])
        elif run[0] == 3:
            tcn_no_tl.append(run[3])
            tcn_tl.append(run[4])

    print(f"CNN: Pre-TL: {sum(cnn_no_tl)/len(cnn_no_tl)} | Post-TL: {sum(cnn_tl)/len(cnn_tl)}")
    print(f"C-LSTM: Pre-TL: {sum(clstm_no_tl)/len(clstm_no_tl)} | Post-TL: {sum(clstm_tl)/len(clstm_tl)}")
    print(f"TCN: Pre-TL: {sum(tcn_no_tl)/len(tcn_no_tl)} | Post-TL: {sum(tcn_tl)/len(tcn_tl)}")


def find_dupe():
    combos = []
    for run in runs:
        if (run[0], run[1], run[2]) not in combos:
            combos.append((run[0], run[1], run[2]))
        else:
            print(f"Dupe at: {run[6]}")

        
if __name__ == '__main__':
    runs = []
    compile_logs()
    print_avail_subj_combs()
    # find_dupe()
    # print_avg_loss()
