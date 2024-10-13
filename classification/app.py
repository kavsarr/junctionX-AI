import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel


model = models.resnet152(pretrained=False)

num_classes = 9
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load('classification\\resnet152_finetuned.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Aluminium', 'Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'Textiles', 'Wood']


class Item(BaseModel):
    image_path: str


app = FastAPI()

@app.get("/")
def read_root():
    return {"Health": "OK"}

@app.post("/predict")
def predict_class(item: Item):

    image_path = item.image_path

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()

    _, predicted_class = torch.max(outputs, 1)

    probability = probabilities[predicted_class.item()]*100
    class_name = class_names[predicted_class.item()]

    print(class_name, probability)
    
    return {"class": class_name,
            "probability": probability}