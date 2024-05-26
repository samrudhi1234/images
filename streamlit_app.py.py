#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit torch torchvision transformers


# In[2]:


import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("Image Caption Generator")
st.write("Upload an image to generate a caption.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for the model
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate the caption
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outputs = model.generate(pixel_values=pixel_values, max_length=16, num_beams=4)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    st.write("Generated Caption:")
    st.write(caption)


# In[ ]:




