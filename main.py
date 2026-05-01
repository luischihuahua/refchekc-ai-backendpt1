import os
import tempfile
import base64
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient

def base64_to_cv2(base64_string):
    image_bytes = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode Roboflow output image.")

    return frame

def cv2_to_base64(frame):
    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        raise ValueError("Could not encode OpenCV image.")

    return base64.b64encode(buffer).decode("utf-8")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # okay for demo; later replace with your Vercel URL
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]

print("API KEY LOADED:", ROBOFLOW_API_KEY)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY,
)

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@app.get("/")
def home():
    return {"status": "Cloud Run backend is working"}

@app.post("/analyze-frame")
async def analyze_frame(
    image: UploadFile = File(...),
    ref_call: str = Form(...)
):
    suffix = os.path.splitext(image.filename)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await image.read())
        image_path = temp.name

    result = client.run_workflow(
        workspace_name="luiss-workspace-kpwcn",
        workflow_id="detect-and-classify",
        images={"image": image_path},
        use_cache=True
    )

    workflow_output = result[0]

    output_image = workflow_output.get("output_image")

    if isinstance(output_image, dict):
        roboflow_image_base64 = output_image.get("value")
    else:
        roboflow_image_base64 = output_image

    # Convert Roboflow output image into OpenCV image
    frame = base64_to_cv2(roboflow_image_base64)

    # Now run your OpenCV processing here
    # Example:
    cv2.putText(
        frame,
        "Processed by OpenCV",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    final_image_base64 = cv2_to_base64(frame)
    cv2.imwrite("debug_output.jpg", frame)
    return {
        "status": "success",
        "ref_call": ref_call,
        "output_image": final_image_base64
    }
