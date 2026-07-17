import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd


# -----------------------------
# Basic settings
# -----------------------------
MODEL_PATH = "MobileNetV2_Model_2.pth"

# Change these if your class order is different
class_names = ["Organic", "Recyclable"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Create the same MobileNetV2 architecture
# -----------------------------
def create_mobilenetv2_model(num_classes=2, dropout=0.4):
    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = create_mobilenetv2_model(
        num_classes=len(class_names),
        dropout=0.4
    )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device
        )

    # Case 1: saved as state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Case 2: saved as full model
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()

    return model


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    # ImageNet normalization, usually used with pretrained MobileNetV2
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image, model):
    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()
    predicted_label = class_names[predicted_index]
    predicted_percentage = probabilities[predicted_index].item() * 100

    all_percentages = {
        class_names[i]: probabilities[i].item() * 100
        for i in range(len(class_names))
    }

    return predicted_label, predicted_percentage, all_percentages


# -----------------------------
# Streamlit interface
# -----------------------------
st.set_page_config(
    page_title="Waste Classification App",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Waste Image Classification")
st.write("Upload an image, and the model will classify it as Organic or Recyclable.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    model = load_model()

    predicted_label, predicted_percentage, all_percentages = predict_image(
        image,
        model
    )

    st.subheader("Prediction Result")

    st.success(
        f"The model predicts: **{predicted_label}** "
        f"with **{predicted_percentage:.2f}%** confidence."
    )

    st.subheader("Class Percentages")

    results_df = pd.DataFrame({
        "Class": list(all_percentages.keys()),
        "Percentage": list(all_percentages.values())
    })

    st.dataframe(results_df, use_container_width=True)

    st.bar_chart(
        results_df.set_index("Class")
    )