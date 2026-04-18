from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI(title="Qlothi Backend")

# Enable CORS so the Chrome extension can make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the fashion segmentation model on startup
print("Loading Segformer fashion model (first run downloads ~350MB)...")
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
fashion_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
fashion_model.eval()
print("Fashion model loaded!")

# Label map for the ATR dataset used by this model
LABEL_MAP = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses",
    4: "upper-clothes", 5: "skirt", 6: "pants", 7: "dress",
    8: "belt", 9: "left-shoe", 10: "right-shoe", 11: "face",
    12: "left-leg", 13: "right-leg", 14: "left-arm", 15: "right-arm",
    16: "bag", 17: "scarf"
}

# Only show these as clickable shopping items
SHOPPABLE_CLASSES = {1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17}
# 1=hat, 3=sunglasses, 4=upper-clothes, 5=skirt, 6=pants, 7=dress,
# 8=belt, 9=left-shoe, 10=right-shoe, 16=bag, 17=scarf

FRIENDLY_NAMES = {
    1: "Hat", 3: "Sunglasses", 4: "Top / Upper Wear",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt",
    9: "Footwear", 10: "Footwear",
    16: "Bag", 17: "Scarf / Accessory"
}

class AnalyzeRequest(BaseModel):
    base64_image: str

@app.post("/analyze")
async def analyze_outfit(request: AnalyzeRequest):
    print(f"Received request with base64 image of length: {len(request.base64_image)}")
    
    try:
        # 1. Decode base64 into PIL Image
        base64_data = request.base64_image
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
            
        image_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        print(f"Image opened: {width}x{height}")
        
        # 2. Run Segformer fashion model
        print("Running fashion segmentation...")
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = fashion_model(**inputs)
        
        # Upsample logits to original image size
        logits = outputs.logits  # shape: (1, num_classes, H, W)
        upsampled = torch.nn.functional.interpolate(
            logits, size=(height, width), mode='bilinear', align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)
        print("Segmentation complete.")
        
        # 3. Extract polygons for each clothing class
        items = []
        for class_id in SHOPPABLE_CLASSES:
            # Create binary mask for this class
            mask = (seg_map == class_id).astype(np.uint8) * 255
            
            # Skip if mask is too small (less than 0.5% of image)
            if np.sum(mask > 0) < (width * height * 0.005):
                continue
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            # Use the largest contour
            largest = max(contours, key=cv2.contourArea)
            
            # Simplify the contour to reduce points (smoother polygon)
            epsilon = 0.005 * cv2.arcLength(largest, True)
            simplified = cv2.approxPolyDP(largest, epsilon, True)
            
            if len(simplified) < 4:
                continue
            
            # Normalize to 0-1 range
            polygon = [[float(pt[0][0]) / width, float(pt[0][1]) / height] for pt in simplified]
            
            # Calculate tight bounding box
            px = [p[0] for p in polygon]
            py = [p[1] for p in polygon]
            bbox = [min(px), min(py), max(px), max(py)]
            
            friendly_name = FRIENDLY_NAMES.get(class_id, LABEL_MAP[class_id])
            
            items.append({
                "id": f"item_{class_id}",
                "class_name": friendly_name,
                "confidence": 0.95,
                "polygon_normalized": polygon,
                "bbox_normalized": bbox
            })
        
        print(f"Successfully extracted {len(items)} clothing items.")
        return {
            "status": "success",
            "message": f"Processed image. Found {len(items)} items.",
            "image_size": {"width": width, "height": height},
            "items": items
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR processing image: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Backend Error: {str(e)}",
            "items": []
        }

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Qlothi API Server</title>
            <style>
                body { font-family: -apple-system, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #f8f9fa; color: #111; }
                .container { text-align: center; padding: 40px; background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
                h1 { font-size: 2rem; margin-bottom: 0.5rem; letter-spacing: -0.5px; }
                p { color: #666; margin-bottom: 2rem; }
                footer { font-size: 14px; color: #888; font-weight: 500; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✨ Qlothi Backend API</h1>
                <p>The AI segmentation engine is online and listening for extension requests.</p>
                <footer>Made with ❤️ by <strong>Kobuilds</strong></footer>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
