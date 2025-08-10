import io
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
from google import genai
from google.genai import types
import gradio as gr
import os

# Configure Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load RFDETR model
CLASSES = {0: "logo"}
model = RFDETRBase(pretrain_weights="weights/checkpoint_best_regular.pth")

# Create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def detect_objects_and_recognize_logos(image, threshold):
    try:
        # Run inference with RFDETR using the provided threshold
        detections = model.predict(image, threshold=threshold)
        
        # Initialize labels for detection and recognition frames
        detection_labels = []
        recognition_labels = []
        brand_names = []
        
        # Process detections and recognize logos with Gemini
        for i, (box, class_id, confidence) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
            class_name = CLASSES[class_id]
            box = [round(i) for i in box.tolist()]  # [x_min, y_min, x_max, y_max]
            
            # Create label for detection frame (class name and confidence only)
            detection_label = f"{class_name} {confidence:.2f}"
            detection_labels.append(detection_label)
            
            # Crop the image using the bounding box
            cropped_image = image.crop((box[0], box[1], box[2], box[3]))
            
            # Convert cropped image to bytes
            img_byte_arr = io.BytesIO()
            cropped_image.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
            
            # Send cropped image to Gemini for logo recognition
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg',
                        ),
                        'Recognize the brand name for the logo and return only the name of the logo. If you don’t know the brand name, return "Unknown"'
                    ])
                brand_name = response.text.strip() if response.text else "Unknown"
            except Exception as e:
                brand_name = f"Gemini Error: {str(e)}"
            
            # Create label for recognition frame (class name, confidence, and brand name)
            recognition_label = f"{class_name} {confidence:.2f} | Brand: {brand_name}"
            recognition_labels.append(recognition_label)
            brand_names.append(brand_name)
            
            # Print detection details
            print(
                f"Detected {class_name} with confidence {round(confidence, 3)} "
                f"at location {box} | Brand: {brand_name}"
            )
        
        # Annotate detection frame (only class name and confidence)
        detection_frame = label_annotator.annotate(
            scene=image.copy(),
            detections=detections,
            labels=detection_labels
        )
        detection_frame = box_annotator.annotate(
            scene=detection_frame.copy(),
            detections=detections
        )
        
        # Annotate recognition frame (class name, confidence, and brand name)
        recognition_frame = label_annotator.annotate(
            scene=image.copy(),
            detections=detections,
            labels=recognition_labels
        )
        recognition_frame = box_annotator.annotate(
            scene=recognition_frame.copy(),
            detections=detections
        )
        
        return detection_frame, recognition_frame, ", ".join([name for name in brand_names if name != "Unknown"])
    
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}", "None"

# Create Gradio interface
interface = gr.Interface(
    fn=detect_objects_and_recognize_logos,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.2, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Detection Frame (RFDETR)"),
        gr.Image(type="pil", label="Recognition Frame (RFDETR + Gemini)"),
        gr.Textbox(label="Detected Brand Names")
    ],
    title="Object Detection and Logo Recognition with RFDETR and Gemini",
    description="Upload an image to detect objects using RFDETR model and recognize logos using Google Gemini. Adjust the confidence threshold to filter detections. Outputs include a detection frame (objects only) and a recognition frame (objects with brand names)."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)