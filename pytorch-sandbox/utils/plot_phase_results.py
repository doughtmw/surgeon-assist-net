import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatch

plt.rcParams.update({'font.size': 18})

phase_names = {
    'Preparation': 0, 
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderPackaging': 4,
    'CleaningCoagulation': 5, 
    'GallbladderRetraction': 6}

def plot_barh(s, n, c, ax, ax_n):
    s_idx = np.where(s == n)
    s_idx = s_idx[0]

    times = zip(s_idx, np.ones(len(list(s_idx))))
    times = list(times)
    ax[ax_n].broken_barh(times, (-1,1), color=c)

    return

def plot_phase_results(save_folder, gt, pr):
    # Open ground truth predictions
    with open(save_folder + gt, 'r') as file:
        gt_dict = {e.strip().split(' ')[0]:e.strip().split(' ')[1] for e in file}

    # Open predictions
    with open(save_folder + pr, 'r') as file:
        pr_dict = {e.strip().split(' ')[0]:e.strip().split(' ')[1] for e in file}

    phase_len = len(pr_dict) - 1
    all_pred = np.array([0])
    all_gt =  np.array([0])

    # Across entirety of video 41
    for x in range(0, 3103):
        # Get the current prediction
        curr_pred = phase_names[pr_dict[str(x)]]
        curr_gt = phase_names[gt_dict[str(x)]]

        # Append the current prediction to numpy array
        all_pred = np.append(all_pred, curr_pred)
        all_gt = np.append(all_gt, curr_gt)
    
    fig, ax = plt.subplots(2)
    c0 = 'tab:blue'
    plot_barh(all_pred, 0, c0, ax, 0)
    plot_barh(all_gt, 0, c0, ax, 1)

    c1 = 'tab:orange'
    plot_barh(all_pred, 1, c1, ax, 0)
    plot_barh(all_gt, 1, c1, ax, 1)

    c2 = 'tab:green'
    plot_barh(all_pred, 2, c2, ax, 0)
    plot_barh(all_gt, 2, c2, ax, 1)

    c3 = 'tab:red'
    plot_barh(all_pred, 3, c3, ax, 0)
    plot_barh(all_gt, 3, c3, ax, 1)

    c4 = 'tab:purple'
    plot_barh(all_pred, 4, c4, ax, 0)
    plot_barh(all_gt, 4, c4, ax, 1)

    c5 = 'tab:pink'
    plot_barh(all_pred, 5, c5, ax, 0)
    plot_barh(all_gt, 5, c5, ax, 1)

    c6 = 'tab:cyan'
    plot_barh(all_pred, 6, c6, ax, 0)
    plot_barh(all_gt, 6, c6, ax, 1)
    
    # Ground truth and LSTM prediction
    ax[0].set(
        # xlabel='Time', 
        ylabel='Pred')
    ax[1].set(
        xlabel='Time', 
        ylabel='GT')

    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])

    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([])

    fc0 = mpatch.Rectangle((0, 0), 1, 1, fc=c0)
    fc1 = mpatch.Rectangle((0, 0), 1, 1, fc=c1)
    fc2 = mpatch.Rectangle((0, 0), 1, 1, fc=c2)
    fc3 = mpatch.Rectangle((0, 0), 1, 1, fc=c3)
    fc4 = mpatch.Rectangle((0, 0), 1, 1, fc=c4)
    fc5 = mpatch.Rectangle((0, 0), 1, 1, fc=c5)
    fc6 = mpatch.Rectangle((0, 0), 1, 1, fc=c6)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        [fc0, fc1, fc2, fc3, fc4, fc5, fc6], 
        ['Preparation', 'CalotTriangleDissection', 
        'ClippingCutting', 'GallbladderDissection', 
        'GallbladderPackaging', 'CleaningCoagulation',
        'GallbladderRetraction'], 
        loc='center right')
    plt.show()

def main():
    save_folder = 'C:/git/surgeon-assist-net/pytorch-sandbox/results/cholec80_256__feat_b0_lite__img_224__len_10__hsize_128__epo_18/'
    gt = 'ground_truth.txt'
    pr = 'pred.txt'
    
    plot_phase_results(save_folder, gt, pr)

if __name__ == "__main__":
    main()