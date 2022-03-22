import os
import torch

import requests
from PIL import Image

import numpy as np
from torchvision import models
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import gradio as gr

# About Segmentation
segmentation_models = {
    "deeplabv3_mobilenet_v3_large": models.segmentation.deeplabv3_mobilenet_v3_large,  
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101, 
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50, 
    "fcn_resnet101": models.segmentation.fcn_resnet101, 
    "fcn_resnet50": models.segmentation.fcn_resnet50, 
    "lraspp_mobilenet_v3_large": models.segmentation.lraspp_mobilenet_v3_large
}
# Preparing About Model Inference
model = segmentation_models["lraspp_mobilenet_v3_large"](pretrained=True, progress=False)
model.eval()

SEGMENTATION_CLASS = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(SEGMENTATION_CLASS)}

def predict(input):
    image = np.array(input)
    image = torch.Tensor(image).type(torch.uint8).permute(2,0,1)
    inp = image.unsqueeze(0)/255.
    inp = F.normalize(inp, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    with torch.no_grad():
        prediction = model(inp)
        normalized_masks = torch.nn.functional.softmax(prediction['out'], dim=1)
        num_classes = normalized_masks.shape[1]
        masks = normalized_masks[0]
        class_dim = 0
        all_classes_masks = masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
        img_with_all_masks = draw_segmentation_masks(image, masks=all_classes_masks, alpha=.6)
        img_with_all_masks = img_with_all_masks.numpy().transpose(1,2,0)
    return img_with_all_masks

img_file_format = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
example_root = "test_img"
example_img_list = [[os.path.join(example_root, file), 0.5] for file in os.listdir(example_root) if file.split(".")[-1].lower() in img_file_format]

gr.Interface(
            title="Segmentation",
            fn=predict, 
            inputs=gr.inputs.Image(type="pil"), 
            outputs=gr.outputs.Image(type="pil"),
            examples= example_img_list,
            flagging_dir="flagged/detection"
            ).launch()