'''
An edited version of the learning curve plot from 'Reinforcement Learning' Assignment 1
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')

        if title:
            self.ax.set_title(title)

    # Add new curve to the plot figure with given label name (optional) with rewards as y values
    def add_curve(self, y, label=None):
        if label:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    # Set the upper/lower bounds of the y-axis
    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    # Save the plot figure given file name
    def save(self, filename='dqn_experiment.png'):
        self.ax.legend()
        self.fig.savefig(filename, dpi=300)


# Use Savitzky-Golayfilter for smoothing given y values
def smooth(y, window, poly=1):
    return savgol_filter(y, window, poly)
