import torch
import torchvision.models as models
import os
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
import json

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_dir = "images"

candidate = [
    "case",
    "machinegun",
    "marksman",
    "music",
    "operator",
    "pistol",
    "rifle",
    "shotgun",
    "sticer",
    "subgun",
]

names = []
for file in os.listdir(images_dir):
    names.append(file)


def eval(model_path, json_path):
    model = torch.load(model_path)
    model.eval()

    predictions = []
    with torch.no_grad():
        for file in tqdm(names):
            image = Image.open(os.path.join(images_dir, file)).convert("RGB")
            img_t = transform(image)
            batch_t = torch.unsqueeze(img_t, 0)
            output = model(batch_t)
            indices = torch.sort(output, descending=True)
            percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
            predict = []
            for i in range(5):
                predict.append(
                    {
                        "score": percentage[indices.indices[0][i]].tolist(),
                        "label": candidate[indices.indices[0][i].tolist()],
                    }
                )
            predictions.append(
                {
                    "image": file,
                    "prediction": predict,
                }
            )
    del model
    with open(json_path, "w") as f:
        json.dump(predictions, f)


eval("models/finetune.pth", "finetune_pred.json")
eval("models/unfreze_params.pth", "unfreze_pred.json")
