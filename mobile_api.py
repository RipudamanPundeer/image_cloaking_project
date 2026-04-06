import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# Import your EXACT working function
from perturbation import generate_adversarial

app = FastAPI(title="Privacy Shield Edge API")

@app.post("/protect")
async def protect_image(
    file: UploadFile = File(...),
    epsilon: float = Form(0.05),
    iters: int = Form(10)
):
    try:
        # 1. Receive image from the UI
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. CRITICAL FIX: Convert to RGB for the PyTorch backend
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 3. Run your verified math engine
        protected_rgb = generate_adversarial(image_rgb, epsilon=epsilon, steps=iters)

        # 4. Convert back to BGR for encoding and sending to the phone
        protected_bgr = cv2.cvtColor(protected_rgb, cv2.COLOR_RGB2BGR)
        _, encoded_img = cv2.imencode('.png', protected_bgr)

        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Listening on all interfaces so a real phone on the same Wi-Fi could connect
    uvicorn.run(app, host="0.0.0.0", port=8000)