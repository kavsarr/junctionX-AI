import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image


model = models.resnet152(pretrained=True)

num_classes = 9
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

state_dict = torch.load('resnet152_finetuned.pth')
model.load_state_dict(state_dict)

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "test_images/plastik.jpg"

image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

class_names = ['Aluminium', 'Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'Textiles', 'Wood']

with torch.no_grad():
    outputs = model(image)

probabilities = torch.nn.functional.softmax(outputs, dim=1)

print(probabilities)

_, predicted_class = torch.max(outputs, 1)

print(class_names[predicted_class.item()])