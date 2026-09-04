import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
model,
train_loader,
val_loader,
device,
epochs,
learning_rate,
weight_decay
):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train_accs = []
    val_accs = []

    best_accuracy = 0


    for epoch in range(epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

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


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = outputs.argmax(
                    dim=1
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


        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: "
            f"{running_loss / len(train_loader):.4f} "
            f"Train: {train_accuracy:.2f}% "
            f"Val: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print(
                "Saved best model!"
            )


    return (
        model,
        train_accs,
        val_accs
    )
