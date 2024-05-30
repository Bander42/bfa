import cv2
import numpy as np
import os
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Load the DINO model
model = load_model("gdino/groundingdino/config/GroundingDINO_SwinT_OGC.py", "gdino/weights/groundingdino_swint_ogc.pth")

# Define thresholds
BOX_THRESHOLD = 0.04
TEXT_THRESHOLD = 0.04

# Define text prompt (shortened for brevity)
text_prompt = "wonder. data. identity. self. personality. consume. resentment. distraction."

# Split the text prompt by ". " to extract classes
classes = text_prompt.split(". ")
class_indices = {class_name: index for index, class_name in enumerate(classes, start=0)}

# Create a directory for temporary files if it doesn't exist
tmp_dir = '/tmp'

def detect_objects(request):
    # Capture live video feed from the default webcam device 0
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        return {"error": "Could not open video stream."}

    while True:
        ret, frame = video_capture.read()
        if not ret:
            return {"error": "Could not read frame."}

        # Save frame as temporary image file
        cv2.imwrite(os.path.join(tmp_dir, 'tmp.jpg'), frame)

        # Load the temporary image file
        tmp_image_path = os.path.join(tmp_dir, 'tmp.jpg')

        try:
            # Load image using load_image function
            image_source, image = load_image(tmp_image_path)

            # Perform object detection using DINO
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=text_prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
        except Exception as e:
            return {"error": f"Error during prediction: {e}"}

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        # Encode the annotated frame as base64 to return it in the response
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Release the video capture object
        video_capture.release()

        # Return the annotated frame in the response
        return {"annotated_frame_base64": annotated_frame_base64}

# To run the function locally, you can simulate a request like this:
# printdetect_objects({})
