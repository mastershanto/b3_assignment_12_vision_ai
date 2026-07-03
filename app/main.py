import io
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image
from app.utils import TakaDetector

app = FastAPI(
    title="Bangladeshi Taka Denomination Detection API",
    description="A REST API serving a custom YOLOv8 model for currency note detection and bounding box prediction.",
    version="1.0.0"
)

# Instantiate detector lazily
detector = None

@app.on_event("startup")
def load_model():
    global detector
    try:
        detector = TakaDetector("model/best.pt")
        print("YOLOv8 Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model at startup: {e}")

@app.get("/")
def read_root():
    return {
        "app": "Bangladeshi Taka Denomination Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Accept image file (JPEG/PNG) and return bounding boxes and denominations"
        }
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_denomination(file: UploadFile = File(...)):
    global detector
    
    # 1. Handle missing files (handled by FastAPI validation, but double-checked)
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing file input."
        )
        
    # 2. Check if the file is a valid image type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        # Fallback file extension check if content-type is missing or general
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        if ext not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format: {file.content_type or ext}. Only JPEG and PNG images are supported."
            )

    try:
        # Read image bytes
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content is empty."
            )
            
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid or corrupt image."
            )
        
        # Instantiate detector if not loaded
        if detector is None:
            detector = TakaDetector("model/best.pt")
            
        # Run prediction
        predictions = detector.predict(image)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "count": len(predictions),
                "predictions": predictions
            }
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        # Return 500 Internal Server Error for unexpected model errors
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "detail": f"An error occurred while processing the image: {str(e)}"
            }
        )

# Import os for filename extension check
import os
