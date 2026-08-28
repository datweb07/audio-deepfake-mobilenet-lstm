import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import config

def plot_history(history, phase="phase1"):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy ({phase})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss ({phase})')
    plt.legend()
    
    plt.savefig(os.path.join(config.PLOTS_DIR, f"training_history_{phase}.png"))
    plt.close()
