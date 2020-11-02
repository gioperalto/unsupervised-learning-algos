import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, name, learner, axes):
        self.name = name
        self.learner = learner
        self.x_axis = axes['x']
        self.y_axis = axes['y']

    def add_plot(self, x, y, label, marker='.'):
        plt.plot(x, y, linestyle='-', marker=marker, label=label)

    def find_max(self, x, y, label):
        max_acc, offset = max(y), max(x)*.0125
        i = y.index(max_acc)
        plt.axvline(x=x[i], label='{}={:.3f} ({})'.format(self.y_axis, y[i], label), color='g')
        plt.text(x=x[i]+offset, y=100, s='{:.1f}'.format(max_acc))

    def find_max_int(self, x, y, label):
        max_acc, offset = max(y), max(x)*.0125
        i = y.index(max_acc)
        plt.axvline(x=x[i], label='{}={} ({})'.format(self.x_axis, x[i], label), color='g')
        plt.text(x=x[i]+offset, y=100, s='{:.1f}'.format(max_acc))

    def find_min(self, x, y, label, top=True):
        min_mse, x_offset = min(y), max(x)*.0125
        y_offset = (max(y)/2.) + min(y) if top else (max(y)/2.) - min(y)
        i = y.index(min_mse)
        plt.axvline(x=x[i], label='{}={:.4f} ({})'.format(self.x_axis, x[i], label), color='g')
        plt.text(x=x[i]+x_offset, y=y_offset, s='{:.5f}'.format(min_mse))

    def find_min_int(self, x, y, label, top=True):
        min_mse, x_offset = min(y), max(x)*.0125
        y_offset = (max(y)/2.) + min(y) if top else min(y) - (max(y)/2.) 
        i = y.index(min_mse)
        plt.axvline(x=x[i], label='{}={} ({})'.format(self.x_axis, x[i], label), color='g')
        plt.text(x=x[i]+x_offset, y=y_offset, s='{:.5f}'.format(min_mse))

    def save(self, loc='best', framealpha=.8, top_limit=None):
        if top_limit is not None:
            plt.ylim(top=top_limit)
        plt.xlabel(self.x_axis)
        plt.ylabel(self.y_axis)
        plt.legend(loc=loc, framealpha=framealpha)
        plt.savefig('images/{}/{}'.format(self.learner, self.name))
        plt.close()