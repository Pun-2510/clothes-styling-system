import random
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import (Dataset, DataLoader)
from torchvision import transforms
from config import *
from model import FashionResNet
from train import train_model

random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

filtered_dataset = dataset.filter(lambda x: x["articleType"] in SELECTED_CLASSES)

if len(filtered_dataset) > MAX_IMAGES:
    indices = random.sample(range(len(filtered_dataset)), MAX_IMAGES)

filtered_dataset = (
    filtered_dataset.select(
        indices
    )
)

print("Number of images:", len(filtered_dataset))

class_names = SELECTED_CLASSES

class_to_idx = {

    name: i

    for i, name in enumerate(
        class_names
    )

}


indices = list(range(len(filtered_dataset)))

labels = [filtered_dataset[i]["articleType"] for i in indices]

train_indices, val_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)

print("Train:",len(train_indices))
print("Validation:",len(val_indices))

train_transform = transforms.Compose([

    transforms.Resize(
        (RESIZE_SIZE, RESIZE_SIZE)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.7, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD

    )
])

val_transform = transforms.Compose([

    transforms.Resize(
        (RESIZE_SIZE, RESIZE_SIZE)
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=NORMALIZE_MEAN,

        std=NORMALIZE_STD

    )

])


class FashionDataset(Dataset):

    def __init__(

        self,

        dataset,

        indices,

        class_to_idx,

        transform=None

    ):

        self.dataset = dataset

        self.indices = indices

        self.class_to_idx = class_to_idx

        self.transform = transform


    def __len__(self):

        return len(
            self.indices
        )


    def __getitem__(

        self,

        idx

    ):

        real_idx = self.indices[idx]

        item = self.dataset[
            real_idx
        ]


        image = item[
            "image"
        ].convert("RGB")


        label_name = item[
            "articleType"
        ]


        label = self.class_to_idx[
            label_name
        ]


        if self.transform:

            image = self.transform(
                image
            )


        return image, label

train_dataset = FashionDataset(filtered_dataset, train_indices, class_to_idx, train_transform)

val_dataset = FashionDataset(filtered_dataset, val_indices, class_to_idx, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n" + "=" * 50)

print("TRAINING BASELINE")

print("=" * 50)

baseline_model = FashionResNet(

num_classes=NUM_CLASSES,

pretrained=False

).to(DEVICE)

baseline_model, baseline_train_accs, baseline_val_accs = train_model(

baseline_model,

train_loader,

val_loader,

DEVICE,

EPOCHS,

LEARNING_RATE,

WEIGHT_DECAY

)


print("\n" + "=" * 50)

print("TRAINING PRETRAINED RESNET18")

print("=" * 50)

model = FashionResNet(

num_classes=NUM_CLASSES,

pretrained=True

).to(DEVICE)

model, train_accs, val_accs = train_model(

model,

train_loader,

val_loader,

DEVICE,

EPOCHS,

LEARNING_RATE,

WEIGHT_DECAY

)

torch.save(

{

    "model_state_dict":
    model.state_dict(),

    "classes":
    class_names,

    "class_to_idx":
    class_to_idx

},

MODEL_PATH

)

print("\n" + "=" * 50)

print(
"MODEL TRAINING COMPLETED"
)

print(
"Saved:",
MODEL_PATH
)

print("=" * 50)
