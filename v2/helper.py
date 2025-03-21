import matplotlib.pyplot as plt
from IPython import display
import numpy as np

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    
    # Add moving average for clarity
    if len(scores) > 10:
        moving_avg = np.convolve(scores, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(scores)), moving_avg, label='Moving Avg (10 games)', linestyle='--')
    
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
