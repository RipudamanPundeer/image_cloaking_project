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
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Run your verified math engine
        # This function handles MTCNN and 160x160 internally
        protected_img = generate_adversarial(image, epsilon=epsilon, steps=iters)

        # 3. Encode and return to UI
        _, encoded_img = cv2.imencode('.png', protected_img)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Listening on all interfaces so a real phone on the same Wi-Fi could connect
    uvicorn.run(app, host="0.0.0.0", port=8000)