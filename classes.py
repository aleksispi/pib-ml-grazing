import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import config
import gc


# Define CNN feature extraction block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        conv_ker = 3
        conv_str = 1
        assert conv_str == conv_ker // 2 and conv_ker % 2 == 1
        pool_ker = 2
        pool_str = 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_ker, padding=conv_str)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str)
        # out_dim_factor below is such that the number of elements of the output tensor
        # with width in_width and height in_height is out_dim_factor * in_width * in_height.
        # Note that the below is specific to the current architecture of the CNNBlock.
        self.out_dim_factor = out_channels * 0.5 * 0.5

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class CNNBlockDelia(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlockDelia, self).__init__()
        
        conv_ker = 3
        conv_str = 1
        assert conv_str == conv_ker // 2 and conv_ker % 2 == 1
        pool_ker = 2
        pool_str = 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=conv_ker, padding=conv_str)       

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
        # out_dim_factor below is such that the number of elements of the output tensor
        # with width in_width and height in_height is out_dim_factor * in_width * in_height.
        # Note that the below is specific to the current architecture of the CNNBlock.
        self.out_dim_factor = out_channels * 0.5 * 0.5

    def forward(self, x):
        x = self.conv1(x)        
        x = self.relu(x)
        
        x = self.conv2(x)       
        x = self.relu(x)
        
        x = self.pool(x)

        return x

# LSTM-based Time Series Classifier
class LSTMClassifier(nn.Module):

    # Set two_branches=True if trying with Sen1+Sen2-models (otherwise assumes just Sen2)
    def __init__(self, in_channels, num_classes, cnn_out_dim, hidden_dim, num_layers, im_height, im_width, bidir=False, two_branches=False):
        super(LSTMClassifier, self).__init__()
        self.cnn = CNNBlock(in_channels, cnn_out_dim)  # Can also use CNNBlockDelia
        lstm_in_dim = int(self.cnn.out_dim_factor * im_height * im_width)
        self.lstm = nn.LSTM(lstm_in_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidir)
        if two_branches:
            self.two_branches = True
            self.cnn2 = CNNBlock(2, cnn_out_dim)
            hidden_dim_lstm2 = hidden_dim // 4  # Set equal?
            self.lstm2 = nn.LSTM(lstm_in_dim, hidden_dim_lstm2, num_layers, batch_first=True, bidirectional=bidir)
            self.fc_in_dim = hidden_dim + hidden_dim_lstm2
            if bidir:
                self.fc_in_dim *= 2
            self.fc = nn.Linear(self.fc_in_dim, num_classes)
        else:
            self.two_branches = False
            if bidir:
                self.fc_in_dim = hidden_dim * 2
            else:
                self.fc_in_dim = hidden_dim
            self.fc = nn.Linear(self.fc_in_dim, num_classes)

    def forward(self, x, seq_lens=None):
        if self.two_branches:
            # In this case, x is a 2-tuple with the first element being the Sentinel-2 data
            # and the second element being the Sentinel-1 data.
            x_sen1 = x[1]
            x = x[0]  # override x with the Sentinel-2 data
            if seq_lens is not None:
                seq_lens_sen1 = seq_lens[1]
                seq_lens = seq_lens[0]
        batch_size, timesteps, C, H, W = x.size()
        # In the below two lines, the input tensor is processed
        # in the "spatial domain" only, i.e. the time dimension
        # is not considered (it is baked into the batch dimension).
        c_in = x.reshape(batch_size * timesteps, C, H, W) 
        c_out = self.cnn(c_in)
        # In the below two lines, the input tensor is processed
        # in the "temporal domain" only, i.e. the spatial dimensions
        # are not considered.
        r_in = c_out.reshape(batch_size, timesteps, -1)

        # LSTM PART
        if seq_lens is not None:
            # Pack the padded input for LSTM processing
            r_in = nn.utils.rnn.pack_padded_sequence(r_in, seq_lens, batch_first=True, enforce_sorted=False)
        # Use LSTM to process the packed input
        r_out, _ = self.lstm(r_in)
        if seq_lens is not None:
            # Unpack the packed output
            r_out, _ = nn.utils.rnn.pad_packed_sequence(r_out, batch_first=True)
        
        # TWO-BRANCH PART
        if self.two_branches:
            _, timesteps_sen1, C, _, _ = x_sen1.size()
            c_in_sen1 = x_sen1.reshape(batch_size * timesteps_sen1, C, H, W)
            c_out_sen1 = self.cnn2(c_in_sen1)
            r_in_sen1 = c_out_sen1.reshape(batch_size, timesteps_sen1, -1)
            if seq_lens_sen1 is not None:
                r_in_sen1 = nn.utils.rnn.pack_padded_sequence(r_in_sen1, seq_lens_sen1, batch_first=True, enforce_sorted=False)
            r_out_sen1, _ = self.lstm2(r_in_sen1)
            if seq_lens_sen1 is not None:
                r_out_sen1, _ = nn.utils.rnn.pad_packed_sequence(r_out_sen1, batch_first=True)
            # Concatenate the last valid outputs of the two branches
            r_out_conc = torch.zeros(batch_size, self.fc_in_dim).to(r_out.device)
            for idx in range(batch_size):
                if seq_lens is None:
                    seq_len = timesteps
                else:
                    seq_len = seq_lens[idx]
                if seq_lens_sen1 is None:
                    seq_len_sen1 = timesteps_sen1
                else:
                    seq_len_sen1 = seq_lens_sen1[idx]
                r_out_conc[idx, :] = torch.cat((r_out[idx, seq_len-1, :], r_out_sen1[idx, seq_len_sen1-1, :]), dim=-1)
            r_out = r_out_conc
            if seq_lens is None:
                r_out = r_out.reshape(batch_size, -1)
                out = self.fc(r_out)
                out_end = out.reshape(batch_size, 1, -1)
                out = torch.zeros(batch_size, timesteps_sen1, 2).to(out.device)
                out[:, -1, :] = out_end
            else:
                r_out = r_out.reshape(batch_size, -1)
                out = self.fc(r_out)
                out = out.reshape(batch_size, 1, -1)
                # To make format with non-sen1 case compatible, we need to ensure that out is a list
                # of length batch_size, where each element is a tensor of shape time_steps x 2.
                # We base this on the seq_lens variable.
                all_outs = []
                for idx, seq_len in enumerate(seq_lens):
                    # Each element in all_outs has shape time_steps x 2, can e.g. be
                    # [23, 2]. But in this sen1 case, we add dummy values for all indices
                    # except the last one.
                    curr_out = torch.zeros(seq_len, 2).to(out.device)
                    curr_out[seq_len - 1, :] = out[idx, :]
                    all_outs.append(curr_out)
                out = all_outs
            # Note that c_out is also returned, as it could be used e.g. for visualizing "spatial attention"
            # etc of model at later stage
            return out, c_out
            
        # Compute output prediction and return it.
        # OBS: This code computes output for each step in the sequence, which can later be used to inspect
        # "intermediate" predictions of the model. The code in train_ml.py however only uses the last
        # for the actual prediction.
        # Reshape the output tensor to be suitable for the fully connected layer
        r_out = r_out.reshape(batch_size * timesteps, -1)
        out = self.fc(r_out)
        out = out.reshape(batch_size, timesteps, -1)
        if seq_lens is not None:
            all_outs = []
            for idx, seq_len in enumerate(seq_lens):
                all_outs.append(out[idx, :seq_len, :]) 
            out = all_outs
        # Note that c_out is also returned, as it could be used e.g. for visualizing "spatial attention"
        # etc of model at later stage
        return out, c_out
        
# Simple 5-layer MLP model (used as basis for cloud optical thickness prediction)
class MLP5(nn.Module):
	def __init__(self, input_dim, output_dim=1, hidden_dim=64, apply_relu=True):
		super(MLP5, self).__init__()
		self.lin1 = nn.Linear(input_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)
		self.lin3 = nn.Linear(hidden_dim, hidden_dim)
		self.lin4 = nn.Linear(hidden_dim, hidden_dim)
		self.lin5 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		self.apply_relu = apply_relu

	def forward(self, x):
		x1 = self.lin1(x)
		x1 = self.relu(x1)
		x2 = self.lin2(x1)
		x2 = self.relu(x2)
		x3 = self.lin3(x2)
		x3 = self.relu(x3)
		x4 = self.lin4(x3)
		x4 = self.relu(x4)
		x5 = self.lin5(x4)
		if self.apply_relu:
			x5[:, 0] = self.relu(x5[:, 0])  # NB: cloud optical thicknesses cannot be negative
		return x5

def replace(string_in, replace_from, replace_to='_'):
    if not isinstance(replace_from, list):
        replace_from = [replace_from]
    string_out = string_in
    for replace_entry in replace_from:
        string_out = string_out.replace(replace_entry, replace_to)
    return string_out


class BaseStat():
    """
    Basic statistic from which all other statistic types inherit
    """
    def __init__(self, name):
        self.name = name
        self.ep_idx = 0
        self.stat_collector = None

    def collect(self, value):
        pass

    def get_data(self):
        return {}

    def next_step(self):
        pass

    def next_ep(self):
        self.ep_idx += 1

    def next_batch(self):
        pass

    def compute_mean(self, mean, value, counter):
        return (counter * mean + value) / (counter + 1)

    def compute_ma(self, ma, value, ma_weight):
        return (1 - ma_weight) * ma + ma_weight * value


class AvgStat(BaseStat):
    """
    Standard average statistic (can track total means, moving averages,
    exponential moving averages etcetera)
    """
    def __init__(self, name, coll_freq='ep', ma_weight=0.001):
        super(AvgStat, self).__init__(name=name)
        self.counter = 0
        self.mean = 0.0
        self.ma = 0.0
        self.last = None
        self.means = []
        self.mas = []
        self.values = []
        self.times = []
        self.coll_freq = coll_freq
        self.ma_weight = ma_weight

    def collect(self, value, delta_counter=1):
        self.counter += delta_counter

        self.values.append(value)
        self.times.append(self.counter)
        self.mean = self.compute_mean(self.mean, value, len(self.means))
        self.means.append(self.mean)
        if self.counter < 10:
            # Want the ma to be more stable early on
            self.ma = self.mean
        else:
            self.ma = self.compute_ma(self.ma, value, self.ma_weight)
        self.mas.append(self.ma)
        self.last = value

    def get_data(self):
        return {'times': self.times, 'means': self.means, 'mas': self.mas, 'values': self.values}

    def print(self, timestamp=None):
        if self.counter <= 0:
            return
        self._print_helper()

    def _print_helper(self, mean=None, ma=None, last=None):

        # Set defaults
        if mean is None:
            mean = self.mean
        if ma is None:
            ma = self.ma
        if last is None:
            last = self.last

        if isinstance(mean, (float, np.floating)):
            print('Mean %-35s tot: %10.5f, ma: %10.5f, last: %10.5f' %
                  (self.name, mean, ma, last))
        else:
            print('Mean %-35s tot:  (%.5f' % (self.name, mean[0]), end='')
            for i in range(1, mean.size - 1):
                print(', %.5f' % mean[i], end='')
            print(', %.5f)' % mean[-1])
            print('%-40s ma:   (%.5f' % ('', ma[0]), end='')
            for i in range(1, ma.size - 1):
                print(', %.5f' % ma[i], end='')
            print(', %.5f)' % ma[-1])
            print('%-40s last: (%.5f' % ('', last[0]), end='')
            for i in range(1, last.size - 1):
                print(', %.5f' % last[i], end='')
            print(', %.5f)' % last[-1])

    def save(self, save_dir):
        file_name = replace(self.name, [' ', '(', ')', '/'], '-')
        file_name = replace(file_name, ['<', '>'], '')
        file_name += '.npz'
        np.savez(os.path.join(save_dir, file_name),
                 values=np.asarray(self.values), means=np.asarray(self.means),
                 mas=np.asarray(self.mas), times=np.asarray(self.times))

    def plot(self, times=None, values=None, means=None, mas=None, save_dir=None):
        # Set defaults
        if times is None:
            times = self.times
        if values is None:
            values = self.values
        if means is None:
            means = self.means
        if mas is None:
            mas = self.mas
        if save_dir is None:
            save_dir_given = None
            save_dir = os.path.join(self.log_dir, 'stats', 'data')
        else:
            save_dir_given = save_dir

        # Define x-label
        if self.coll_freq == 'ep':
            xlabel = 'episode'
        elif self.coll_freq == 'step':
            xlabel = 'step'

        if np.asarray(values).ndim > 1:
            # Plot all values
            self._plot(times, values, self.name + ' all', xlabel, 'y', None,
                       save_dir_given)

            # Plot total means
            self._plot(times, means, self.name + ' total mean', xlabel, 'y', None,
                       save_dir_given)

            # Plot moving averages
            self._plot(times, mas, self.name + ' total exp ma', xlabel, 'y', None,
                       save_dir_given)
        else:
            self._plot_in_same(times, [values, means, mas],
                               self.name, xlabel, 'y',
                               ['all-data', 'mean', 'ma'],
                               [None, '-.', '-'], [0.25, 1.0, 1.0],
                               save_dir_given)

        # Also save current data to file
        if save_dir_given is None:
            file_name = replace(self.name, [' ', '(', ')', '/'], '-')
            file_name = replace(file_name, ['<', '>'], '')
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, file_name), values)

    def _plot(self, x, y, title='plot', xlabel='x', ylabel='y', legend=None,
              log_dir=None):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()

    def _plot_in_same(self, x, ys, title='plot', xlabel='x', ylabel='y',
                      legend=None, line_styles=None, alphas=None,
                      log_dir=None):
        if alphas is None:
            alphas = [1.0 for _ in range(len(ys))]
        plt.figure()
        for i in range(len(ys)):
            if line_styles[i] is not None:
                plt.plot(x, ys[i],
                         linestyle=line_styles[i], alpha=alphas[i])
            else:
                plt.plot(x, ys[i], 'yo', alpha=alphas[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend is None:
            plt.legend([str(k) for k in range(np.asarray(y).shape[1])])
        else:
            plt.legend(legend)
        title_to_save = replace(title, [' ', '(', ')', '/'], '-')
        title_to_save = replace(title_to_save, ['<', '>'], '')
        if log_dir is None:
            log_dir = os.path.join(self.log_dir, 'stats', 'plots')
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        plt.savefig(os.path.join(log_dir, title_to_save + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()


class StatCollector():
    """
    Statistics collector class
    """
    def __init__(self, log_dir, tot_nbr_steps, print_iter):
        self.stats = OrderedDict()
        self.log_dir = log_dir
        self.ep_idx = 0
        self.step_idx = 0
        self.epoch_idx = 0
        self.print_iter = print_iter
        self.tot_nbr_steps = tot_nbr_steps

    def has_stat(self, name):
        return name in self.stats

    def register(self, name, stat_info, ma_weight=0.001):
        if self.has_stat(name):
            sys.exit("Stat already exists")

        if stat_info['type'] == 'avg':
            stat_obj = AvgStat(name, stat_info['freq'], ma_weight=ma_weight)
        else:
            sys.exit("Stat type not supported")

        stat = {'obj': stat_obj, 'name': name, 'type': stat_info['type']}
        self.stats[name] = stat

    def s(self, name):
        return self.stats[name]['obj']

    def next_step(self):
        self.step_idx += 1

    def next_ep(self):
        self.ep_idx += 1
        for _, stat in self.stats.items():
            stat['obj'].next_ep()
        if self.ep_idx % self.print_iter == 0:
            self.print()
            self._plot_to_hdock()

    def print(self):
        for _, stat in self.stats.items():
            stat['obj'].print()

    def plot(self):
        for _, stat in self.stats.items():
            stat['obj'].plot(save_dir=self.log_dir)

    def save(self):
        for _, stat in self.stats.items():
            stat['obj'].save(save_dir=self.log_dir)