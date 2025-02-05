import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Define emotions and CNN model
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionCNN(torch.nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, len(EMOTIONS))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Function to load the model
def load_model(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model

# Function to predict emotion
def predict_emotion(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return EMOTIONS[predicted.item()]

# Streamlit App Layout
st.sidebar.title("Emotion Detection App")
page = st.sidebar.selectbox("Navigation", ["Home", "Emotion Detection"])

if page == "Home":
    st.title("Welcome to the Emotion Detection App")
    st.write("""
    ### Real-World Relevance
    Emotion detection technology has significant applications in several fields, including:
    
    **1. Healthcare**  
    - *Mental Health Monitoring:* Emotion detection can assist therapists in tracking patients' emotions, identifying signs of stress, anxiety, or depression.  
    - *Autism Support:* Helping individuals with autism recognize emotions in others, improving social interactions.  
    - *Telemedicine:* Enhancing remote diagnoses by analyzing patient emotions during video consultations.  
    
    **2. Education**  
    - *Student Engagement:* Monitor students' emotions to identify whether they are engaged, confused, or bored, allowing teachers to adapt teaching methods.  
    - *Personalized Learning:* Tailoring lesson difficulty based on students' emotional states.  
    - *Anti-Bullying Measures:* Detecting distress in students through emotion analysis to address bullying incidents in real-time.  
    
    **3. Customer Service**  
    - *Customer Feedback:* Analyzing customer emotions during interactions to gauge satisfaction levels.  
    - *Improved Support:* Chatbots equipped with emotion detection can respond empathetically, improving user experience.  
    - *Real-Time Insights for Staff:* Helping customer service representatives adapt to customer emotions for better interactions.  
    
    This technology holds immense potential but requires careful consideration of ethical issues, such as privacy, data security, and bias mitigation.
    """)
    st.write("Navigate to the **Emotion Detection** page to try out the app.")

elif page == "Emotion Detection":
    st.title("Emotion Detection from Images")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load the model and predict emotion
        model = load_model("emotion_cnn_model.pth")
        emotion = predict_emotion(model, uploaded_file)
        st.write(f"Predicted Emotion: {emotion}")
