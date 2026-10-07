import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """CLIP-style contrastive objective with same-class images as positives."""
    if embeddings.ndim != 2 or len(embeddings) < 2:
        return embeddings.sum() * 0.0
    logits = embeddings @ embeddings.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positives.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    logits = logits.masked_fill(identity, float("-inf"))
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_counts = positives.sum(dim=1).clamp_min(1)
    losses = -(log_probabilities.masked_fill(~positives, 0.0).sum(dim=1)
               / positive_counts)
    return losses[valid].mean()

def train_model(
model,
train_loader,
val_loader,
device,
epochs,
learning_rate,
weight_decay,
best_model_path=None,
label_smoothing=0.0,
early_stopping_patience=0,
lr_patience=2,
lr_factor=0.3,
min_learning_rate=1e-6,
gradient_clip_norm=0.0,
contrastive_weight=0.0,
contrastive_temperature=0.07,
):

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor,
        patience=lr_patience, min_lr=min_learning_rate
    )

    train_accs = []
    val_accs = []

    best_accuracy = 0
    best_state = None
    epochs_without_improvement = 0


    for epoch in range(epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if contrastive_weight > 0:
                outputs, embeddings = model(images, return_embedding=True)
            else:
                outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )
            if contrastive_weight > 0:
                loss = loss + contrastive_weight * supervised_contrastive_loss(
                    embeddings, labels, contrastive_temperature
                )

            loss.backward()

            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()


            running_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()


        train_accuracy = (
            100 * correct / total
        )

        train_accs.append(
            train_accuracy
        )

        model.eval()

        val_correct = 0
        val_total = 0
        correct_by_class = None
        total_by_class = None


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = outputs.argmax(
                    dim=1
                )

                if total_by_class is None:
                    num_classes = outputs.shape[1]
                    total_by_class = torch.zeros(num_classes, dtype=torch.long)
                    correct_by_class = torch.zeros(num_classes, dtype=torch.long)

                labels_cpu = labels.detach().cpu()
                predictions_cpu = predictions.detach().cpu()
                total_by_class += torch.bincount(labels_cpu, minlength=num_classes)
                correct_by_class += torch.bincount(
                    labels_cpu[predictions_cpu == labels_cpu], minlength=num_classes
                )

                val_total += labels.size(0)

                val_correct += (
                    predictions == labels
                ).sum().item()


        val_accuracy = (
            100 * val_correct / val_total
        )

        val_accs.append(
            val_accuracy
        )

        represented = total_by_class > 0
        macro_val_accuracy = 100.0 * (
            correct_by_class[represented].float()
            / total_by_class[represented].float()
        ).mean().item()
        scheduler.step(macro_val_accuracy)
        current_lr = optimizer.param_groups[0]["lr"]


        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: "
            f"{running_loss / len(train_loader):.4f} "
            f"Train: {train_accuracy:.2f}% "
            f"Val: {val_accuracy:.2f}% "
            f"Macro Val: {macro_val_accuracy:.2f}% "
            f"LR: {current_lr:.2e}"
        )

        if macro_val_accuracy > best_accuracy:

            best_accuracy = macro_val_accuracy
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if best_model_path is not None:
                torch.save(best_state, best_model_path)

            print(
                "Saved best model!"
            )
        else:
            epochs_without_improvement += 1
            if (
                early_stopping_patience > 0
                and epochs_without_improvement >= early_stopping_patience
            ):
                print("Early stopping: macro validation accuracy did not improve.")
                break


    if best_state is not None:
        model.load_state_dict(best_state)

    return (
        model,
        train_accs,
        val_accs,
        best_accuracy,
    )
