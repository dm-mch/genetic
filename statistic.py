import re
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class StatisticFeaturesPlot:
    
    def __init__(self, features, interactive=True):
        self.features = features
        self.length = len(features)
        self.groups = self.make_groups()
        if interactive:
            plt.ion()
            
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.bars = self.ax.bar(np.arange(self.length), [0] * self.length)
        self.ax.set_xticks([g[1] for g in self.groups])
        self.ax.set_xticklabels([g[0] for g in self.groups], rotation="vertical")

        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        #elf.ax.grid()
        self.texts = []
        
    def make_groups(self):
        groups = []
        r = re.compile(r"^([a-zA-Z_]+)([0-9]+)")
        current = ''
        for i,fname in enumerate(self.features):
            g = re.findall(r, fname)[0]
            if g[0][:-1] != current:
                groups.append((g[0][:-1], i))
                current = g[0][:-1]
        return groups

    def plot(self, ydata):
        #Update data (with the new _and_ the old points)
        #self.lines.set_xdata(xdata)
        for i in range(self.length):
            self.bars[i].set_height(ydata[i])

        # clear text
        for t in self.texts:
            t.remove()
        self.texts = []
        max10_indexs = np.argsort(ydata)[::-1][:int(self.length*0.20)]
        for bar_i in max10_indexs:
            height = ydata[bar_i]
            self.texts.append(self.ax.text(self.bars[bar_i].get_x(), height, self.features[bar_i]))

        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


    def plot_population(self, population, feature_gen='ts_features'):
        features = []
        for p in population:
            features.append(p.entity.get_genom()[feature_gen].data)
        self.plot(np.stack(features).sum(axis=0))
