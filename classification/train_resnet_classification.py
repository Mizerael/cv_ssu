import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

NUM_CLASSES = 10

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = ImageFolder("dataset/train", transform=transform)
val_dataset = ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    model_path,
):
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects.double() / len(val_dataset)

        print(
            "Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}".format(
                epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc
            )
        )
    torch.save(model, model_path)

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT,
)

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=5,
    device=device,
    model_path="models/finetune.pth",
)

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT,
)

for param in model.parameters():
    param.requires_grad = True

model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=5,
    device=device,
    model_path="models/unfreze_params.pth",
)
