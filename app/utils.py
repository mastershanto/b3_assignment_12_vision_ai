import os
from PIL import Image

# pyrefly: ignore [missing-import]
from ultralytics import YOLO

class TakaDetector:
    def __init__(self, model_path: str = "model/best.pt"):
        # Default fallback to runs if not found in model/
        if not os.path.exists(model_path):
            alternative_path = os.path.join("runs", "currency_train", "weights", "best.pt")
            if os.path.exists(alternative_path):
                model_path = alternative_path
            else:
                # Fallback to yolo11n.pt if best.pt is not available (for robust failure recovery)
                print(f"Warning: model weights {model_path} not found. Falling back to default yolo11n.pt")
                model_path = "yolo11n.pt"
                
        self.model = YOLO(model_path)
        
    def predict(self, image: Image.Image):
        # Run YOLO inference
        results = self.model(image)
        predictions = []
        
        # Parse prediction results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class label name
                class_id = int(box.cls[0])
                name = result.names[class_id]
                
                # Confidence score
                confidence = float(box.conf[0])
                
                # Bounding box [xmin, ymin, xmax, ymax]
                xyxy = box.xyxy[0].tolist()
                
                predictions.append({
                    "denomination": name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "xmin": round(xyxy[0], 2),
                        "ymin": round(xyxy[1], 2),
                        "xmax": round(xyxy[2], 2),
                        "ymax": round(xyxy[3], 2)
                    }
                })
        return predictions
