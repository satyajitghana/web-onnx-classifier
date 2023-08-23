import onnxruntime as ort
import numpy as np
from PIL import Image

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

with open("imagenet_classes.txt", "r") as f:
    classes_response = f.read()
classes_list = [line.strip() for line in classes_response.split('\n')]

ort_session = ort.InferenceSession("resnetv2_50.onnx")
output = ort_session.run(
    ['output'], {'input': np.random.randn(1, 3, 224, 224).astype(np.float32)})

print(f"random_output = {output}")

img = Image.open("test_image.jpeg")
img = img.convert("RGB")
img = img.resize((224, 224))
img_np = np.array(img)

print(f"image shape = {img_np.shape}")

img_np = img_np / 255.0
img_np = (img_np - mean) / std
img_np = img_np.transpose(2, 0, 1)

ort_outputs = ort_session.run(
    ['output'], {'input': img_np[None, ...].astype(np.float32)})

pred_class_idx = np.argmax(ort_outputs[0])

predicted_class = classes_list[pred_class_idx]

print(f"{predicted_class=}")
