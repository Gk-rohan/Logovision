import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

CLASSES = {0: "logo"}

model = RFDETRBase(pretrain_weights="checkpoint_best_regular.pth", device='cpu')

image =  Image.open("SCR-20250528-jvug-2.jpeg")
detections = model.predict(image, threshold=0.14)

labels = [
    f"{CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

annotated_image.save("annotated_output.jpeg")