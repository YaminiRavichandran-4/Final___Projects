import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# ✅ Load class labels
class_names = sorted(os.listdir(r"C:\Users\YAMINI RAVICHANDRAN\OneDrive\Desktop\emotion\plant\archive (1)\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"))

# ✅ Define Model (same as before)
class PlantDiseaseCNN(nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(class_names))
        )

    def forward(self, X):
        X = self.conv_layers(X)
        X = self.fc_layers(X)
        return X

# ✅ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantDiseaseCNN().to(device)
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=device))
model.eval()

# ✅ Define image transformation
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload an image of a leaf, and the model will classify its disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]

    st.success(f"Prediction: **{class_name}**")
