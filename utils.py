
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def init_log(log_path):
    """Create a log textfile, overwrite the old one if it already exists
    
    Arguments:
        log_path {str} -- Filename of the log textfile
    """

    textline = "\t".join([
        'epoch', 
        'loss', 
        'accuracy',
        ])
    with open(log_path, 'w') as data:
        data.write("".join([textline, "\n"]))

def write_log(log_path, epoch, loss, accuracy):
    """Write data into existing logfile
    
    Arguments:
        log_path {str} -- Filename of the log textfile
        epoch {int} -- epoch to log
        loss {float} -- loss value to log
        accuracy {float} -- accuracy value to log
    """
    textline = '\t'.join([
        '{:d}'.format(epoch),
        '{:g}'.format(loss),
        '{:g}'.format(accuracy),
        ])
    with open(log_path, 'a') as data:
        data.write("".join([textline, "\n"]))

def plot_loss(train_filename, test_filename):
    """Plot the loss curve during training for train and test set
    
    Arguments:
        train_filename {str} -- Filename of the train log textfile
        test_filename {str} -- Filename of the test log textfile
    """

    #load loss data from disk
    with open(train_filename, 'r') as f:
        train_loss = np.genfromtxt(f, names=True, usecols=['epoch', 'loss'], delimiter='\t')
        train_loss = train_loss.view(float).reshape(train_loss.shape + (-1,))
    with open(test_filename, 'r') as f:
        test_loss = np.genfromtxt(f, names=True, usecols=['epoch', 'loss'], delimiter='\t')
        test_loss = test_loss.view(float).reshape(test_loss.shape + (-1,))

    #plot loss data
    plot_curves([train_loss, test_loss], x_label='epochs', y_label='loss', labels=['train', 'test'])

def plot_accuracy(train_filename, test_filename):
    """Plot the accuracy curve during training for train and test set
    
    Arguments:
        train_filename {str} -- Filename of the train log textfile
        test_filename {str} -- Filename of the test log textfile
    """

    #load accuracy data from disk
    with open(train_filename, 'r') as f:
        train_accuracy = np.genfromtxt(f, names=True, usecols=['epoch', 'accuracy'], delimiter='\t')
        train_accuracy = train_accuracy.view(float).reshape(train_accuracy.shape + (-1,))
    with open(test_filename, 'r') as f:
        test_accuracy = np.genfromtxt(f, names=True, usecols=['epoch', 'accuracy'], delimiter='\t')
        test_accuracy = test_accuracy.view(float).reshape(test_accuracy.shape + (-1,))

    #plot loss data
    plot_curves([train_accuracy, test_accuracy], x_label='epochs', y_label='accuracy', labels=['train', 'test'])

def plot_curves(data_list, x_label='x', y_label='y', labels=[]):
    """generic plot function for drawing lines
    
    Arguments:
        data_list {list} -- List of ndarrays to plot
    
    Keyword Arguments:
        x_label {str} -- label of x axis (default: {'x'})
        y_label {str} -- label of y axis (default: {'y'})
        labels {list} -- label of plotted lines (default: {[]})
    
    Returns:
        figure -- matplotlib figure
        ax -- matplotlib ax
    """

    if labels:
        label_iterator = iter(labels)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    handles = []
    legend_labels = []

    for i, data in enumerate(data_list):
        handle, = ax.plot(data[:,0], data[:,1], linestyle='-')
        handles.append(handle)   
        if labels:
            legend_labels.append(next(label_iterator))
        else:
            legend_labels.append(i)

    plt.legend(handles, legend_labels)
    xlabels = ax.get_xticklabels()
    plt.setp(xlabels, rotation=45, horizontalalignment='right')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(0.,1.)
    plt.grid()
    fig.tight_layout()

    return fig, ax


def update_mean_std(existingAggregate, newValue):
    """Welford's Online algorithm for computing mean and std of a distribution online.
    mean accumulates the mean of the entire dataset.
    m2 aggregates the squared distance from the mean.
    count aggregates the number of samples seen so far.

    Arguments:
        existingAggregate {tuple} -- Intermediate results (count, mean, m2)
        newValue {float} -- A new value drawn from the distribution

    Returns:
        tuple -- updated aggregate (count, mean, m2)
    """

    (count, mean, m2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    m2 += delta * delta2

    return (count, mean, m2)

def finalize_mean_std(existingAggregate):
    """Retrieve the mean, variance and sample variance from an aggregate

    Arguments:
        existingAggregate {tuple} -- Intermediate results (count, mean, m2)
        
    Returns:
        tuple -- distribution statistics: (mean, standard deviation, standard deviation with sample normalization
    """

    (count, mean, m2) = existingAggregate
    (mean, variance, sample_variance) = (mean, m2/count, m2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, np.sqrt(variance), np.sqrt(sample_variance))


class AddGaussianNoise(object):
    """
    Add Gaussian noise to PIL image
    """

    def __init__(self, blend_alpha_range):
        self.blend_alpha_range = blend_alpha_range

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        w, h = img.size
        if img.mode == 'RGB':
            c = 3
        elif img.mode == 'L':
            c = 1
        else:
            raise ValueError('Got image of unknown mode')
        noise_image = np.random.normal(0., 1., (h, w, c))
        noise_image = Image.fromarray(noise_image, img.mode)
        blend_alpha = np.random.random()*(self.blend_alpha_range[1] - self.blend_alpha_range[0]) + self.blend_alpha_range[0]

        return Image.blend(img, noise_image, blend_alpha)

    def __repr__(self):
        return self.__class__.__name__ + '(blend_alpha={0})'.format(self.blend_alpha_range)