# ============================================================
# BERT FASHION CLASSIFICATION - TRAIN
# ============================================================
#
# Dataset:
# https://huggingface.co/datasets/nreimers/fashion-dataset
#
# Input:
#     productDisplayName
#
# Label:
#     articleType
#
# Output:
#     ./bert_fashion_model
#
# ============================================================


# ============================================================
# CUDA MEMORY CONFIG
# ============================================================

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================
# LIBRARIES
# ============================================================

import json
import numpy as np
import torch

from datasets import load_dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


# ============================================================
# CONFIG
# ============================================================

from config import (
    MODEL_PATH,
    BASE_MODEL,
    DATASET_NAME,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    MAX_SAMPLES,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_LENGTH,
    SEED,
)


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("DEVICE")
print("=" * 60)

if torch.cuda.is_available():

    print("CUDA available")
    print("GPU:", torch.cuda.get_device_name(0))

    DEVICE = "cuda"

else:

    print("CUDA not available")
    print("Training on CPU")

    DEVICE = "cpu"

print()


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

print("Dataset:", DATASET_NAME)
print("Loading...")

dataset = load_dataset(DATASET_NAME)

data = dataset["train"]

print()
print("Original samples:", len(data))
print("Columns:", data.column_names)
print()


# ============================================================
# CHECK COLUMNS
# ============================================================

if TEXT_COLUMN not in data.column_names:

    raise ValueError(
        f"Không tìm thấy cột '{TEXT_COLUMN}'. "
        f"Các cột hiện tại: {data.column_names}"
    )


if LABEL_COLUMN not in data.column_names:

    raise ValueError(
        f"Không tìm thấy cột '{LABEL_COLUMN}'. "
        f"Các cột hiện tại: {data.column_names}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

print("=" * 60)
print("CLEAN DATA")
print("=" * 60)

data = data.filter(
    lambda x:
        x[TEXT_COLUMN] is not None
        and x[LABEL_COLUMN] is not None
        and str(x[TEXT_COLUMN]).strip() != ""
        and str(x[LABEL_COLUMN]).strip() != ""
)

print("Samples after cleaning:", len(data))
print()


# ============================================================
# CREATE LABEL MAPPING
# ============================================================

print("=" * 60)
print("CREATING LABEL MAPPING")
print("=" * 60)

labels = sorted(
    list(
        set(
            str(x).strip()
            for x in data[LABEL_COLUMN]
            if x is not None
        )
    )
)

num_labels = len(labels)

label2id = {
    label: idx
    for idx, label in enumerate(labels)
}

id2label = {
    idx: label
    for idx, label in enumerate(labels)
}

print("Number of classes:", num_labels)

for idx, label in id2label.items():

    print(
        f"{idx:3d} -> {label}"
    )

print()


# ============================================================
# CONVERT LABEL
# ============================================================

def convert_label(example):

    label = str(
        example[LABEL_COLUMN]
    ).strip()

    example["label"] = label2id[label]

    return example


data = data.map(
    convert_label
)


# ============================================================
# CREATE TEXT
# ============================================================

def create_text(example):

    example["text"] = str(
        example[TEXT_COLUMN]
    ).strip()

    return example


data = data.map(
    create_text
)


# ============================================================
# LIMIT DATASET
# ============================================================

if MAX_SAMPLES is not None:

    if MAX_SAMPLES < len(data):

        print("=" * 60)
        print("LIMIT DATASET")
        print("=" * 60)

        print(
            "Requested samples:",
            MAX_SAMPLES
        )

        # ----------------------------------------------------
        # Stratified sampling
        # ----------------------------------------------------

        indices = np.arange(len(data))

        labels_array = np.array(
            data["label"]
        )

        try:

            selected_indices, _ = train_test_split(
                indices,
                train_size=MAX_SAMPLES,
                random_state=SEED,
                shuffle=True,
                stratify=labels_array
            )

        except ValueError:

            print(
                "Warning: stratified sampling failed."
            )

            print(
                "Using random sampling instead."
            )

            rng = np.random.default_rng(
                SEED
            )

            selected_indices = rng.choice(
                indices,
                size=MAX_SAMPLES,
                replace=False
            )

        data = data.select(
            selected_indices.tolist()
        )

        print(
            "Samples used:",
            len(data)
        )

        print()

    else:

        print(
            "MAX_SAMPLES >= dataset size."
        )

        print(
            "Using full dataset:",
            len(data)
        )

else:

    print("=" * 60)
    print("FULL DATASET")
    print("=" * 60)

    print(
        "Using all samples:",
        len(data)
    )

print()


# ============================================================
# KEEP ONLY NECESSARY COLUMNS
# ============================================================

data = data.select_columns([
    "text",
    "label"
])


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)

# Convert labels for stratification
labels_for_split = np.array(
    data["label"]
)

indices = np.arange(
    len(data)
)

try:

    train_indices, eval_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=labels_for_split
    )

except ValueError:

    print(
        "Warning: stratified split failed."
    )

    print(
        "Using random split."
    )

    train_indices, eval_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )


train_dataset = data.select(
    train_indices.tolist()
)

eval_dataset = data.select(
    eval_indices.tolist()
)


print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(eval_dataset)
)

print()


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)

label_mapping = {

    "label2id": label2id,

    "id2label": {
        str(k): v
        for k, v in id2label.items()
    },

    "num_labels": num_labels
}


labels_path = os.path.join(
    MODEL_PATH,
    "labels.json"
)


with open(
    labels_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        label_mapping,
        f,
        ensure_ascii=False,
        indent=4
    )


print(
    "Saved labels:",
    labels_path
)

print()


# ============================================================
# TOKENIZER
# ============================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

print(
    "Base model:",
    BASE_MODEL
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

print()


# ============================================================
# TOKENIZATION
# ============================================================

print("=" * 60)
print("TOKENIZING")
print("=" * 60)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


print("Tokenization completed.")
print()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING BERT MODEL")
print("=" * 60)

print(
    "Model:",
    BASE_MODEL
)

print(
    "Classes:",
    num_labels
)


model = AutoModelForSequenceClassification.from_pretrained(

    BASE_MODEL,

    num_labels=num_labels,

    label2id=label2id,

    id2label=id2label,

    ignore_mismatched_sizes=True
)


print()
print("Model loaded.")
print()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels_true = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels_true,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            labels_true,

            predictions,

            average="weighted",

            zero_division=0
        )
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("=" * 60)
print("TRAINING CONFIG")
print("=" * 60)

print(
    "Epochs:",
    NUM_EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Effective batch size:",
    BATCH_SIZE *
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Max length:",
    MAX_LENGTH
)

print(
    "FP16:",
    torch.cuda.is_available()
)

print()


training_args = TrainingArguments(

    output_dir="./training_output",

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=1,

    # KHÔNG load checkpoint tốt nhất vào cuối train
    load_best_model_at_end=False,

    logging_strategy="steps",

    logging_steps=50,

    fp16=torch.cuda.is_available(),

    seed=SEED,

    push_to_hub=False,

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# START TRAINING
# ============================================================

print()
print("=" * 60)
print("START TRAINING")
print("=" * 60)

print()

trainer.train()

# ============================================================
# FINAL EVALUATION
# ============================================================

print()
print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

metrics = trainer.evaluate()

for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )

print()


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 60)
print("SAVING MODEL")
print("=" * 60)

trainer.save_model(
    MODEL_PATH
)

tokenizer.save_pretrained(
    MODEL_PATH
)


# Save labels

with open(
    labels_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        label_mapping,
        f,
        ensure_ascii=False,
        indent=4
    )


print()
print("MODEL SAVED SUCCESSFULLY")
print(
    "Path:",
    os.path.abspath(MODEL_PATH)
)