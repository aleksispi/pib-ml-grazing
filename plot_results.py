import os, sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Global vars
LOG_PATH_DATE = '2025-07-06_10-40-23'


MA_SMOOTH = 0.015
MIN_Y = 0.5
MAX_Y = 1.01
MAX_Y_VAL = 1
MAX_X = None
STAT_NAME = 'Accuracy'  # 'Accuracy', 'Accuracy_grazing' or 'Accuracy_no_activity'

def _custom_ma(data, ma_smooth=MA_SMOOTH):
    for idx, val in enumerate(data['values']):
        if idx < 2:
            data['mas_custom'][idx] = data['means'][idx]
        else:
            data['mas_custom'][idx] = (1 - ma_smooth) * data['mas_custom'][idx - 1] + ma_smooth * data['values'][idx]

def _plot(datas, title='', xlabel='# train steps', ylabel=STAT_NAME, start_it=0, max_x=None, min_y=None,
          max_y=None, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1)
    else:
        ax = fig.add_subplot(1, 2, 2)
    legend_entries = []
    for idx, data in enumerate(datas):
        # So either each data in datas can be a (1) a list of data, or (2) a list of list of data.
        # The below code currently assumes case (1).
        # But we want to add support for case (2) as well, and in that case, we want to compute the mean and std of the data in data,
        # and plot this mean and associated std (as a shaded area around the mean).
        legend_entries.append(idx)
        if type(data[0]) is dict:  # case (1)
            x = data[0]['times']
            y = data[0]['mas_custom']
            plt.plot(x[start_it:], y[start_it:])
        else:  # case (2)
            assert type(data[0]) is list
            # Now go over all the data in data, and compute mean and std.
            len_y_min = np.min([len(data_i[0]['mas_custom']) for data_i in data])
            y_all = np.zeros((len(data), len_y_min))
            for i, data_i in enumerate(data):
                x = data_i[0]['times']
                y = data_i[0]['mas_custom']
                y_all[i] = y[:len_y_min]
            y_mean = np.mean(y_all, axis=0)
            y_std = np.std(y_all, axis=0)
            x = x[:len_y_min]
            plt.plot(x[start_it:], y_mean[start_it:])
            plt.fill_between(x[start_it:], y_mean[start_it:] - y_std[start_it:], y_mean[start_it:] + y_std[start_it:], alpha=0.2)
            legend_entries.append(idx)
    print("nbr-data", len(y[start_it:]), "min-err", np.min(y[start_it:]))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(legend_entries)
    ax = plt.gca()
    if max_x is None:
        max_x = x[-1]
    if max_y is None:
        max_y = max(np.max(y['means'][start_it:]), np.max(y['mas'][start_it:]))
    if min_y is None:
        min_y = min(np.min(y['means'][start_it:]), np.min(y['mas'][start_it:]))
    ax.set_xlim([0, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_aspect(max_x / (max_y-min_y))
    return fig

# Read data from log paths
seed_paths = [seed_path for seed_path in os.listdir(os.path.join('log', LOG_PATH_DATE)) if '.py' not in seed_path]
log_path_dates = [os.path.join('log', LOG_PATH_DATE, seed_path) for seed_path in seed_paths]
L2_losses_all = []
L2_losses_all_val = []
for log_path_date in log_path_dates:
    log_path = os.path.join(log_path_date, 'train_stats', STAT_NAME + '.npz')
    L2_losses = np.load(log_path)
    L2_losses = {'means': L2_losses['means'], 'mas': L2_losses['mas'],
                 'values': L2_losses['values'], 'times': L2_losses['times'],
                 'mas_custom': np.zeros_like(L2_losses['mas'])}
    log_path = os.path.join(log_path_date, 'train_stats', STAT_NAME + '_val.npz')
    L2_losses_val = np.load(log_path)
    L2_losses_val = {'means': L2_losses_val['means'], 'mas': L2_losses_val['mas'],
                     'values': L2_losses_val['values'], 'times': [10 * vv for vv in L2_losses_val['times']],
                     'mas_custom': np.zeros_like(L2_losses_val['mas'])}

    # Create MA-smoothing of raw data
    _custom_ma(L2_losses)
    _custom_ma(L2_losses_val, ma_smooth=10*MA_SMOOTH)

    # Add to list of all
    L2_losses_all.append([L2_losses, -1])
    L2_losses_all_val.append([L2_losses_val, -1])

# Plot results
fig_out = _plot(L2_losses_all, title='train', max_x=MAX_X, min_y=MIN_Y, max_y=MAX_Y, fig=None)
_plot(L2_losses_all_val, title='val', min_y=MIN_Y, max_y=MAX_Y_VAL, fig=fig_out)

fig_out.savefig('result_plot.png')
#fig_out.savefig('result_plot.eps')
plt.cla()
plt.clf()
plt.close('all')
print("Saved result plot!")