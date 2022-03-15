import os
import torch

import requests
from PIL import Image
from torchvision import transforms

import gradio as gr

# Preparing About Model Inference
model = torch.hub.load('pytorch/vision:v0.12.0', 'efficientnet_b0', pretrained=True).eval()

with open("./labels.txt") as f:
    labels = f.read().split("\n")

def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}    
    return confidences

img_file_format = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
example_root = "test_img"
example_img_list = [os.path.join(example_root, file) for file in os.listdir(example_root) if file.split(".")[-1].lower() in img_file_format]

gr.Interface(
            title="Image Classification",
            fn=predict, 
            inputs=gr.inputs.Image(type="pil"),
            outputs=gr.outputs.Label(num_top_classes=3),
            examples= example_img_list,
            flagging_dir="flagged/classification"
            ).launch()