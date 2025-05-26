import random
import numpy as np
import torch
import matplotlib.pyplot as plt
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_training_results(hist_pretrained, hist_scratch):
    plt.figure(figsize=(10, 5))
    # plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, len(hist_pretrained)+1), hist_pretrained, label="Pretrained")
    plt.plot(range(1, len(hist_scratch)+1), hist_scratch, label="Scratch")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, len(hist_pretrained)+1, 5))
    plt.legend()
    plt.savefig('validation_accuracy_comparison.png')
    plt.show()