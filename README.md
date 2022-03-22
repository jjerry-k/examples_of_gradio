# examples_of_gradio

Gradio를 이용한 Deep learning model 데모 예제!
모두 PyTorch 를 이용한 예제입니다! 
Custom model을 이용한 예제도 제작 예정입니다!

## Setup
``` bash
conda create -n gradio python=3.8 
conda activate gradio
pip install torch torchvision gradio jinja2 Pillow
```

1. Simple Gradio Example [코드](01_ui.py)

2. Simple Image Classification Example Using PyTorch Model [코드](02_classification.py)

3. Simple Object Detection Example Using PyTorch Model [코드](03_object_detection.py)

3. Simple Image Segmentation Example Using PyTorch Model [코드](04_segmentation.py)