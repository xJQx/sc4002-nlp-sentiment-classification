import matplotlib.pyplot as plt
import os

def save_graph(save_filename_prefix: str = None, dataset_type: str=None):
    
    if save_filename_prefix.startswith("/"): save_filename_prefix = save_filename_prefix[1:]
    
    # Create save directory if does not exist
    if save_filename_prefix:
        dir_name = os.path.dirname(save_filename_prefix)
        if dir_name != "" and dir_name != "/":
            if not os.path.isdir(f"plots/{dir_name}"):
                os.makedirs(f"plots/{dir_name}")
                
    if save_filename_prefix:
        plt.savefig(f"plots/{save_filename_prefix}_{dataset_type}_loss_accuracy.png")

                
    return

def plot_loss_acc_graph(
        loss_list: list[float], 
        acc_list: list[float],
        dataset_type: str,
        subtitle: str,
        save_filename_prefix: str = None,
        display: bool =True
    ):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    epoch_counts = [c for c in range(1, len(loss_list)+1)]

    # Overall Title for the figure
    fig.suptitle(f"({dataset_type}) {subtitle}", fontsize=16)

    # Loss
    ax1.plot(epoch_counts, loss_list)
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Epoch Number")
    ax1.grid(True)

    # Accuracy
    ax2.plot(epoch_counts, acc_list, color="red")
    ax2.set_xlabel("Epoch Number")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Epoch Number")
    ax2.grid(True)
    
    save_graph(f"plots/{save_filename_prefix}_{dataset_type}_loss_accuracy.png", dataset_type)
    
    if display:
        plt.show()

    # Reset plot
    plt.close()
    
    
def plot_multiple_loss_acc_graph(
    loss_list: list[list[float]],
    acc_list: list[list[float]],
    label_list: list,
    dataset_type: str,
    subtitle: str,
    save_filename_prefix: str = None,
    display: bool =True
):

    labels = ['max', 'avg', 'last']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Overall Title for the figure
    fig.suptitle(f"({dataset_type}) {subtitle}", fontsize=16, y=1.0)

    # Loss
    for i, label in zip(loss_list, labels): 
        ax1.plot(range(1, 1+len(i)), i, label=label)  
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Epoch Number")
    ax1.grid(True)
    ax1.legend()

    # Accuracy
    for i, label in zip(acc_list, labels): 
        ax2.plot(range(1, 1+len(i)), i, label=label)  
    ax2.set_xlabel("Epoch Number")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Epoch Number")
    ax2.grid(True)
    ax2.legend()

    save_graph(f"plots/{save_filename_prefix}_{dataset_type}_loss_accuracy.png", dataset_type)
    
    if display:
        plt.show()

    # Reset plot
    plt.close()