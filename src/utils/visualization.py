from matplotlib import pyplot as plt

def show_plots(history):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(10,10))
    plt.xlabel("Epochs")
    ax1.set_title("Loss")

    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.legend()

    ax2.set_title("Accuracy")
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='validation')
    ax2.legend()