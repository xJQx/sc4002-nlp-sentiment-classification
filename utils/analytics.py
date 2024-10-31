import re
import subprocess
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

import wandb


def load_tensorboard_logs(log_dir):
    """
    Load the final epoch metrics from TensorBoard logs.

    Parameters:
    - log_dir (str): The root directory containing TensorBoard log files.

    Returns:
    - pd.DataFrame: A DataFrame with final epoch metrics.
    """
    metrics = ["val_loss", "val_acc", "train_loss", "train_acc", "epoch"]

    log_dir = Path(log_dir)
    log_file_names = list(log_dir.rglob("events.out.tfevents*"))

    results = []

    # Regex pattern to capture metadata from the filename
    pattern = (
        r"batch_size_(\d+)-lr_([\deE.-]+)-optimizer_(\w+)-hidden_dim_(\d+)"
        r"-num_layers_(\d+)-sentence_representation_type_(\w+)"
    )

    for log_path in log_file_names:
        event_acc = event_accumulator.EventAccumulator(str(log_path))
        event_acc.Reload()

        # Initialize a dictionary to store final metric values and metadata
        final_epoch_data = {metric: None for metric in metrics}
        final_epoch_data["filename"] = log_path.name

        # Extract metadata from filename
        match = re.search(pattern, str(log_path))

        if match:
            final_epoch_data["batch_size"] = int(match.group(1))
            final_epoch_data["learning_rate"] = float(match.group(2))
            final_epoch_data["optimizer_name"] = match.group(3)
            final_epoch_data["hidden_dim"] = int(match.group(4))
            final_epoch_data["num_layers"] = int(match.group(5))
            final_epoch_data["sentence_representation_type"] = match.group(6)
        else:
            print(f"Filename pattern does not match for {log_path}")

        for metric in metrics:
            if metric in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(metric)

                if metric == "val_acc" and events:
                    max_val_acc = max([event.value for event in events])
                    final_epoch_data["val_acc"] = max_val_acc
                else:
                    final_event = events[-1] if events else None
                    if final_event:
                        final_epoch_data[metric] = final_event.value

        results.append(final_epoch_data)

    df = pd.DataFrame(results)

    column_order = [
        "val_acc",
        "train_acc",
        "batch_size",
        "hidden_dim",
        "learning_rate",
        "optimizer_name",
        "num_layers",
        "sentence_representation_type",
        "epoch",
        "train_loss",
        "val_loss",
        "filename",
    ]
    return df[column_order]


def get_config_from_file(log_path):
    pattern = (
        r"batch_size_(\d+)-lr_([\deE.-]+)-optimizer_(\w+)-hidden_dim_(\d+)"
        r"-num_layers_(\d+)-sentence_representation_type_(\w+)"
    )

    config = {}
    match = re.search(pattern, str(log_path))

    if match:
        config["batch_size"] = int(match.group(1))
        config["learning_rate"] = float(match.group(2))
        config["optimizer_name"] = match.group(3)
        config["hidden_dim"] = int(match.group(4))
        config["num_layers"] = int(match.group(5))
        config["sentence_representation_type"] = match.group(6)

    return config


def upload_to_wandb(
    log_dir: str = "tb_logs/",
    project: str = "nlp_proj",
    entity: str = "jinghua",
):
    wandb.login()
    tensorboard_root_dir = Path(log_dir)

    for run_dir in tensorboard_root_dir.rglob("batch_size*"):
        config = get_config_from_file(run_dir.name)

        run = wandb.init(
            project=project,
            entity=entity,
            name=run_dir.name,
            config=config,
        )
        run_id = run.id
        run.finish()

        subprocess.run(
            ["wandb", "sync", run_dir, "-p", project, "-e", entity, "--id", run_id]
        )

        print(f"Uploaded TensorBoard logs from: {run_dir}")
