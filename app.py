import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model (InceptionV3 finetuned)
MODEL_PATH = 'inception_v3_finetuned_L2.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Function to prepare the image before passing to the model
def prepare_image(img):
    img = img.resize((150, 150))  # Resize image to fit InceptionV3 input shape
    img_array = np.array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image
    return img_array

# Define the Streamlit interface
def main():
    st.title("InceptionV3 Fine-Tuned Model Prediction")

    st.write("Upload an image to get the model's prediction.")

    # File uploader widget for image input
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Open and display the uploaded image
        img = Image.open(uploaded_image)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Prepare the image
        prepared_img = prepare_image(img)

        # Get the model prediction
        prediction = model.predict(prepared_img)

        # Show the prediction results
        st.write("Prediction Result:")
        st.write(prediction)

        # Optionally, you can interpret the result if you have class labels
        # For example, assuming the model predicts a 1000-class ImageNet model
        class_labels = [
            "Alopecia Areata",
            "Contact Dermatitis",
            "Folliculitis",
            "Head Lice",
            "Healthy Hair",
            "Lichen Planus",
            "Male Pattern Baldness",
            "Psoriasis",
            "Seborrheic Dermatitis",
            "Telogen Effluvium",
            "Tinea Capitis"
        ]
  # Replace with actual labels
        predicted_class = np.argmax(prediction, axis=1)  # Get the index of the max probability
        st.write(f"Predicted class: {class_labels[predicted_class[0]]}")

if __name__ == '__main__':
    main()