import re
import subprocess
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from datasets import Dataset
from lightning import LightningModule
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.data import DataLoader

_RNN_LOG_FILE_PATTERN = (
    r"batch_size_(\d+)-lr_([\deE.-]+)-optimizer_(\w+)-hidden_dim_(\d+)"
    r"-num_layers_(\d+)-sr_type_(\w+)-freeze_(\w+)"
)

_RNN_LOG_FILE_PATTERN_2 = (
    r"batch_size_(\d+)-lr_([\deE.-]+)-optimizer_(\w+)-hidden_dim_(\d+)"
    r"-num_layers_(\d+)-sr_type_(\w+)-freeze_(\w+)-rnn_type_(\w+)-bidirectional_(\w+)"
)

_CNN_LOG_FILE_PATTERN = (
    r"batch_size_(\d+)-lr_([\deE.-]+)-optimizer_(\w+)-hidden_dim_(\d+)"
    r"-n_grams_((?:\d+_?)+)-dropout_([\deE.-]+)"
)
_CNN_LOG_FILE_PATTERN_2 = (
    r"batch_size_(\d+)-lr_([\deE.-]+)-optimizer_(\w+)-hidden_dim_(\d+)"
)

_METRICS = ["val_loss", "val_acc", "train_loss", "train_acc", "epoch"]


def test_top_n_models(
    df: pd.DataFrame,
    model_classifier: LightningModule,
    test_dataset: Dataset,
    n: int = 20,
):
    result_df = df.head(n).copy().reset_index(drop=True)
    result_df["test_loss"] = None
    result_df["test_acc"] = None

    for index, row in result_df.iterrows():
        filename = row["filename"]
        matched_files = list(Path().rglob(filename))

        if not matched_files:
            print(f"Model checkpoint not found! {filename}")
            continue

        checkpoint_dir = matched_files[0].parent / "checkpoints"
        checkpoint_files = (
            list(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.exists() else []
        )

        if not checkpoint_files:
            print(f"No checkpoint files found in the checkpoint directory! {filename}")
            continue

        model = model_classifier.load_from_checkpoint(checkpoint_files[0])

        test_dataloader = DataLoader(test_dataset)
        trainer = L.Trainer(accelerator="cpu")
        results = trainer.test(model, test_dataloader)

        if results:
            result_df.at[index, "test_loss"] = results[0].get("test_loss", None)
            result_df.at[index, "test_acc"] = results[0].get("test_acc", None)

    column_order = ["test_acc", "test_loss"] + [
        col for col in result_df.columns if col not in ["test_loss", "test_acc"]
    ]

    return result_df[column_order]


def load_tensorboard_logs(log_dir):
    """
    Load the final epoch metrics from TensorBoard logs.

    Parameters:
    - log_dir (str): The root directory containing TensorBoard log files.

    Returns:
    - pd.DataFrame: A DataFrame with final epoch metrics.
    """

    log_dir = Path(log_dir)
    log_file_names = list(log_dir.rglob("events.out.tfevents*"))

    results = []

    for log_path in log_file_names:
        event_acc = event_accumulator.EventAccumulator(str(log_path))
        event_acc.Reload()

        # Extract metadata from filename
        data = match_rnn_log(log_path) or match_cnn_log(log_path)

        if not data:
            print(f"Filename pattern does not match for {log_path}")

        data["filename"] = log_path.name

        for metric in _METRICS:
            if metric in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(metric)

                if metric == "val_acc" and events:
                    max_val_acc = max([event.value for event in events])
                    data["val_acc"] = max_val_acc
                elif metric == "val_loss" and events:
                    min_val_loss = min([event.value for event in events])
                    data["val_loss"] = min_val_loss
                else:
                    final_event = events[-1] if events else None
                    if final_event:
                        data[metric] = final_event.value

        results.append(data)

    df = pd.DataFrame(results)
    df = df.dropna(axis=0, subset=["val_acc", "val_loss"])

    column_order = [
        "val_acc",
        "batch_size",
        "hidden_dim",
        "learning_rate",
        "optimizer_name",
    ]

    last_cols = [
        "epoch",
        "val_loss",
        "filename",
    ]
    column_order += [
        col for col in df.columns if col not in column_order and col not in last_cols
    ]

    column_order += last_cols

    return df[column_order]


def load_tensorboard_logs_from_huggingface_trainer(log_dir):
    """
    Load the final epoch metrics from TensorBoard logs generated by HuggingFace Trainer.

    Parameters:
    - log_dir (str): The root directory containing TensorBoard log files.

    Returns:
    - pd.DataFrame: A DataFrame with final epoch metrics.
    """
    metrics = ["eval/loss", "eval/accuracy"]

    log_dir = Path(log_dir)
    log_file_names = list(log_dir.rglob("events.out.tfevents*"))

    results = []

    for log_path in log_file_names:
        event_acc = event_accumulator.EventAccumulator(str(log_path))
        event_acc.Reload()

        # Initialize a dictionary to store final metric values and metadata
        data = {metric: None for metric in metrics}
        data["filename"] = log_path.name
        data["model"] = log_path.parent.name

        if "eval/loss" in event_acc.Tags()["scalars"]:
            data["val_loss"] = min(
                [event.value for event in event_acc.Scalars("eval/loss")]
            )
        if "eval/accuracy" in event_acc.Tags()["scalars"]:
            data["val_acc"] = max(
                [event.value for event in event_acc.Scalars("eval/accuracy")]
            )

        results.append(data)

    df = pd.DataFrame(results)

    column_order = [
        "model",
        "val_acc",
        "val_loss",
        "filename",
    ]
    return df[column_order]


def load_tensorboard_logs_from_huggingface_trainer(log_dir):
    """
    Load the final epoch metrics from TensorBoard logs generated by HuggingFace Trainer.

    Parameters:
    - log_dir (str): The root directory containing TensorBoard log files.

    Returns:
    - pd.DataFrame: A DataFrame with final epoch metrics.
    """
    metrics = ["eval/loss", "eval/accuracy"]

    log_dir = Path(log_dir)
    log_file_names = list(log_dir.rglob("events.out.tfevents*"))

    results = []

    for log_path in log_file_names:
        event_acc = event_accumulator.EventAccumulator(str(log_path))
        event_acc.Reload()

        # Initialize a dictionary to store final metric values and metadata
        data = {metric: None for metric in metrics}
        data["filename"] = log_path.name
        data["model"] = log_path.parent.name

        if "eval/loss" in event_acc.Tags()["scalars"]:
            data["val_loss"] = min(
                [event.value for event in event_acc.Scalars("eval/loss")]
            )
        if "eval/accuracy" in event_acc.Tags()["scalars"]:
            data["val_acc"] = max(
                [event.value for event in event_acc.Scalars("eval/accuracy")]
            )

        results.append(data)

    df = pd.DataFrame(results)

    column_order = [
        "model",
        "val_acc",
        "val_loss",
        "filename",
    ]
    return df[column_order]


def get_config_from_file(log_path):
    pattern = (
        r"batch_size_(\d+)-lr_([\deE.-]+)-optimizer_(\w+)-hidden_dim_(\d+)"
        r"-num_layers_(\d+)-sr_type_(\w+)-freeze_(\w+)"
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
        config["freeze"] = match.group(7) == "True"

    return config


def get_result_from_file(log_path):
    log_dir = Path(log_path)
    log_file_names = list(log_dir.rglob("events.out.tfevents*"))

    event_acc = event_accumulator.EventAccumulator(str(log_file_names[0]))
    event_acc.Reload()

    metrics = ["val_loss", "val_acc", "train_loss", "train_acc", "epoch"]

    final_epoch_data = {metric: None for metric in metrics}

    for metric in metrics:
        if metric in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(metric)

            if metric == "val_acc" and events:
                max_val_acc = max([event.value for event in events])
                final_epoch_data["val_acc"] = max_val_acc
            elif metric == "val_loss" and events:
                min_val_loss = min([event.value for event in events])
                final_epoch_data["val_loss"] = min_val_loss
            else:
                final_event = events[-1] if events else None
                if final_event:
                    final_epoch_data[metric] = final_event.value

    return final_epoch_data


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


def match_rnn_log(log_path: str):
    match = re.search(_RNN_LOG_FILE_PATTERN, str(log_path)) or re.search(
        _RNN_LOG_FILE_PATTERN_2, str(log_path)
    )

    data = {metric: None for metric in _METRICS}
    if not match:
        return {}
    data["batch_size"] = int(match.group(1))
    data["learning_rate"] = float(match.group(2))
    data["optimizer_name"] = match.group(3)
    data["hidden_dim"] = int(match.group(4))
    data["num_layers"] = int(match.group(5))
    data["sentence_representation_type"] = match.group(6)
    data["freeze"] = match.group(7) == "True"
    if len(match.groups()) == 9:
        data["rnn_type"] = match.group(8)
        data["bidirectional"] = match.group(9)

    return data


def match_cnn_log(log_path: str):
    match = re.search(_CNN_LOG_FILE_PATTERN, str(log_path)) or re.search(
        _CNN_LOG_FILE_PATTERN_2, str(log_path)
    )

    data = {metric: None for metric in _METRICS}

    if match:
        data["batch_size"] = int(match.group(1))
        data["learning_rate"] = float(match.group(2))
        data["optimizer_name"] = match.group(3)
        data["hidden_dim"] = int(match.group(4))
        if len(match.groups()) == 6:
            data["n_grams"] = match.group(5)
            data["dropout"] = match.group(6)
        else:
            data["n_grams"] = "3_4_5"
            data["dropout"] = 0.3

    if not match:
        return {}

    return data


def plot_val_acc(log_filename: str):
    if not log_filename:
        raise ValueError("log_filename cannot be empty or None.")

    matches = list(Path().rglob(log_filename))
    if not matches:
        raise FileNotFoundError(f"No log file found matching {log_filename}.")

    event_acc = event_accumulator.EventAccumulator(str(matches[0]))
    event_acc.Reload()

    if "val_acc" not in event_acc.Tags()["scalars"]:
        raise KeyError("'val_acc' not found in the log file.")

    steps = []
    val_acc = []
    for event in event_acc.Scalars("val_acc"):
        steps.append(event.step)
        val_acc.append(event.value)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=steps, y=val_acc, marker="o")
    plt.title("Validation Accuracy Across Steps", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.grid(True)
    plt.show()
