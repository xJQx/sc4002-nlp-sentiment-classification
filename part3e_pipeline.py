"""Pipeline for running part 3.5
Example usage:
```bash
python part3_5_pipeline.py --model roberta-finetuned
```
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parents[1]))

import argparse

import torch
from datasets import DatasetDict, load_dataset
from torchmetrics import Accuracy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


def train():
    parser = argparse.ArgumentParser(description="Improvements")

    parser.add_argument(
        "--model",
        choices=["roberta", "roberta-finetuned", "gpt2", "gpt2-finetuned", "t5"],
        default="roberta",
        help="Model to use",
    )

    args = parser.parse_args()

    print(args)

    # set up model
    # encoder only (pretrained)
    if args.model == "roberta":
        model_name = "roberta-base"
    # encoder only (finetuned)
    elif args.model == "roberta-finetuned":
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    # decoder only (pretrained)
    elif args.model == "gpt2":
        model_name = "gpt2"
    # decoder only (finetuned)
    elif args.model == "gpt2-finetuned":
        model_name = "mnoukhov/gpt2-imdb-sentiment-classifier"
    # encoder-decoder
    elif args.model == "t5":
        model_name = "google-t5/t5-small"
    else:
        raise NotImplementedError

    run_name = f"{args.model}"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name  # , num_labels=2  # binary classification
    )
    if model.config.pad_token_id == None:
        model.config.pad_token_id = model.config.eos_token_id

    # prepare dataset
    dataset = load_dataset("rotten_tomatoes")
    tokenized_datasets = tokenize(dataset, model_name)

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # create trainer
    trainer = CustomTrainer(
        run_name=run_name,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # train
    trainer.train()

    pred = trainer.predict(test_dataset)
    print("Test metrics:")
    print(pred.metrics)

    return trainer


# default optimizer: AdamW
training_args = TrainingArguments(
    output_dir="./part3_5_logs",  # output directory of results
    num_train_epochs=6,  # number of train epochs
    report_to="tensorboard",  # log to tensorboard
    eval_strategy="steps",  # check evaluation metrics at each epoch
    logging_steps=10,  # we will log every 10 steps
    eval_steps=200,  # we will perform evaluation every 200 steps
    save_steps=200,  # we will save the model every 200 steps
    save_total_limit=1,  # we only save the last checkpoint or the best one (the last one might be the best one)
    load_best_model_at_end=True,  # we will load the best model at the end of training
    metric_for_best_model="accuracy",  # metric to see which model is better
    #### effective batch_size = per_device_train_batch_size x gradient_accumulation_steps ####
    #### We set effective batch_size to 2048 (8 x 256) ####
    per_device_train_batch_size=int(16 / torch.cuda.device_count()),
    per_device_eval_batch_size=int(16 / torch.cuda.device_count()),
    # gradient_accumulation_steps=256,
)


class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        run_name: str,
        **kwargs,
    ):
        training_args.logging_dir = f"{training_args.output_dir}/runs/{run_name}"
        super().__init__(
            *args, compute_metrics=compute_metrics, args=training_args, **kwargs
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Override the default compute_loss.
        Use Cross Entropy Loss for multiclass classification (>= 2).
        """
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute cross entropy loss
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred: EvalPrediction):
    """
    Compute metrics using torchmetrics for a given set of predictions and labels.

    Args:
    pred (EvalPrediction): An object containing model predictions and labels.

    Returns:
    dict: A dictionary containing metric results.
    """
    # Extract labels and predictions
    labels = pred.label_ids
    preds = pred.predictions

    # for t5 model, the predictions is in the form of a tuple with the logits as the only element in the tuple
    if isinstance(preds, tuple):
        preds = preds[0]

    num_classes = preds.shape[1]

    # Convert to torch tensors
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)

    # Initialize metrics
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(
        torch.cuda.current_device()
    )

    # Calculate metrics (automatically does argmax)
    accuracy_score = accuracy(preds, labels)

    # Convert to CPU for serialization
    return {
        "accuracy": accuracy_score.cpu().item(),
    }


def tokenize(dataset: DatasetDict, tokenizer_name: str, input_col_name: str = "text"):
    def _tokenize(examples):
        return tokenizer(
            examples[input_col_name],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = (
        dataset.map(_tokenize, batched=True)
        .select_columns(["input_ids", "attention_mask", "label"])
        .with_format("torch")
    )
    return tokenized_datasets


if __name__ == "__main__":
    train()
