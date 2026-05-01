import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://refcheck-ai-cyan.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY,
)


def box_to_bounds(center_x, center_y, width, height):
    left = center_x - width / 2
    right = center_x + width / 2
    top = center_y - height / 2
    bottom = center_y + height / 2
    return left, right, top, bottom


def is_ball_in_strike_zone(ball_center, strike_zone):
    zone_x = strike_zone["x"]
    zone_y = strike_zone["y"]
    zone_width = strike_zone["width"]
    zone_height = strike_zone["height"]

    ball_x = ball_center["x"]
    ball_y = ball_center["y"]

    strike_zone_area = zone_width * zone_height

    left, right, top, bottom = box_to_bounds(
        zone_x,
        zone_y,
        zone_width,
        zone_height
    )

    inside_x = left <= ball_x <= right
    inside_y = top <= ball_y <= bottom
    is_strike = inside_x and inside_y

    if is_strike:
        verdict = "STRIKE"
        reason = (
            f"The ball marker center is at ({ball_x}, {ball_y}), which is inside the strike zone. "
            f"The strike zone area is {strike_zone_area} px². "
            f"The x-coordinate {ball_x} is within [{left}, {right}], "
            f"and the y-coordinate {ball_y} is within [{top}, {bottom}]."
        )
    else:
        verdict = "BALL"

        reasons = []

        if not inside_x:
            reasons.append(
                f"The x-coordinate {ball_x} is outside the strike-zone x-range [{left}, {right}]."
            )

        if not inside_y:
            reasons.append(
                f"The y-coordinate {ball_y} is outside the strike-zone y-range [{top}, {bottom}]."
            )

        reason = (
            f"The ball marker center is at ({ball_x}, {ball_y}), which is outside the strike zone. "
            f"The strike zone area is {strike_zone_area} px². "
            + " ".join(reasons)
        )

    return {
        "verdict": verdict,
        "strike_zone_area_px2": strike_zone_area,
        "strike_zone_bounds": {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
        },
        "ball_center": {
            "x": ball_x,
            "y": ball_y,
        },
        "reason": reason,
    }


def normalize_ref_call(ref_call: str):
    cleaned = ref_call.strip().upper()

    if cleaned not in ["BALL", "STRIKE"]:
        raise HTTPException(
            status_code=400,
            detail="ref_call must only be 'ball' or 'strike'."
        )

    return cleaned


def extract_predictions(workflow_result):
    if isinstance(workflow_result, list):
        workflow_output = workflow_result[0]
    else:
        workflow_output = workflow_result

    if "predictions" in workflow_output:
        predictions = workflow_output["predictions"]

        if isinstance(predictions, dict) and "predictions" in predictions:
            return predictions["predictions"]

        if isinstance(predictions, list):
            return predictions

    for value in workflow_output.values():
        if isinstance(value, dict) and "predictions" in value:
            return value["predictions"]

        if isinstance(value, list):
            possible_predictions = [
                item for item in value
                if isinstance(item, dict) and "x" in item and "y" in item
            ]

            if possible_predictions:
                return possible_predictions

    return []


def get_best_prediction(predictions, target_names):
    matches = []

    for pred in predictions:
        class_name = str(pred.get("class", "")).lower().strip()

        if class_name in target_names:
            matches.append(pred)

    if not matches:
        return None

    return max(matches, key=lambda p: p.get("confidence", 0))


@app.get("/")
def home():
    return {
        "status": "Cloud Run backend is working",
        "frontend": "https://refcheck-ai-cyan.vercel.app"
    }


@app.post("/analyze-frame")
async def analyze_frame(
    image: UploadFile = File(...),
    ref_call: str = Form(...)
):
    ref_verdict = normalize_ref_call(ref_call)

    suffix = os.path.splitext(image.filename)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await image.read())
        image_path = temp.name

    try:
        ball_result = client.run_workflow(
            workspace_name="luiss-workspace-kpwcn",
            workflow_id="detect-and-classify",
            images={"image": image_path},
            use_cache=True
        )

        ball_predictions = extract_predictions(ball_result)

        ball_marker = get_best_prediction(
            ball_predictions,
            target_names={
                "ball_marker",
                "ball marker",
                "ball",
                "marker"
            }
        )

        if ball_marker is None:
            return {
                "verdict": "bad call",
                "explanation": (
                    "No ball marker was detected in the image, so the system could not verify "
                    "whether the referee call was correct."
                )
            }

        ball_center = {
            "x": ball_marker["x"],
            "y": ball_marker["y"],
            "width": ball_marker.get("width"),
            "height": ball_marker.get("height"),
        }

        strike_zone_result = client.run_workflow(
            workspace_name="luiss-workspace-kpwcn",
            workflow_id="detect-and-classify-2",
            images={"image": image_path},
            use_cache=True
        )

        strike_zone_predictions = extract_predictions(strike_zone_result)

        strike_zone = get_best_prediction(
            strike_zone_predictions,
            target_names={
                "strike_zone",
                "strike zone",
                "zone"
            }
        )

        if strike_zone is None:
            return {
                "verdict": "bad call",
                "explanation": (
                    "The ball marker was detected, but no strike-zone bounding box was detected. "
                    "The system could not complete the call verification."
                )
            }

        strike_zone_box = {
            "x": strike_zone["x"],
            "y": strike_zone["y"],
            "width": strike_zone["width"],
            "height": strike_zone["height"],
        }

        image_analysis = is_ball_in_strike_zone(
            ball_center,
            strike_zone_box
        )

        detected_verdict = image_analysis["verdict"]

        is_good_call = detected_verdict == ref_verdict

        if is_good_call:
            final_verdict = "good call"

            if detected_verdict == "STRIKE":
                explanation = (
                    f"Good call. The referee called a strike, and the image-based analysis also "
                    f"detected a strike. {image_analysis['reason']} Therefore, the ball was inside "
                    f"the strike-zone bounding box."
                )
            else:
                explanation = (
                    f"Good call. The referee called a ball, and the image-based analysis also "
                    f"detected a ball. {image_analysis['reason']} Therefore, the ball was outside "
                    f"the strike-zone bounding box."
                )

        else:
            final_verdict = "bad call"

            if ref_verdict == "BALL" and detected_verdict == "STRIKE":
                explanation = (
                    f"Bad call. The referee called a ball, but the image-based analysis detected "
                    f"a strike. {image_analysis['reason']} The analysis shows that the ball was "
                    f"inside the strike-zone bounding box."
                )
            else:
                explanation = (
                    f"Bad call. The referee called a strike, but the image-based analysis detected "
                    f"a ball. {image_analysis['reason']} The analysis shows that the ball was "
                    f"outside the strike-zone bounding box."
                )

        return {
            "verdict": final_verdict,
            "explanation": explanation
        }

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
