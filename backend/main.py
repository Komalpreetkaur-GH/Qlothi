from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import base64
import os
import time
import requests
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

# ATR label map produced by mattmdjaga/segformer_b2_clothes
LABEL_MAP = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses",
    4: "upper-clothes", 5: "skirt", 6: "pants", 7: "dress",
    8: "belt", 9: "left-shoe", 10: "right-shoe", 11: "face",
    12: "left-leg", 13: "right-leg", 14: "left-arm", 15: "right-arm",
    16: "bag", 17: "scarf"
}

# Synthetic id: left-shoe (9) + right-shoe (10) are merged into one "Footwear" category
# so the user doesn't get two differently-labeled dots for one pair of shoes.
FOOTWEAR_CLASS = 100
SHOE_CLASSES = (9, 10)

# Classes the user can actually shop for.
SHOPPABLE_CLASSES = (1, 3, 4, 5, 6, 7, 8, FOOTWEAR_CLASS, 16, 17)

FRIENDLY_NAMES = {
    1: "Hat", 3: "Sunglasses", 4: "Top / Upper Wear",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt",
    FOOTWEAR_CLASS: "Footwear",
    16: "Bag", 17: "Scarf / Accessory"
}

# Small accessories otherwise get dropped on full-body pins where they occupy very little area.
SMALL_ITEM_CLASSES = {1, 3, 8}  # hat, sunglasses, belt
MIN_AREA_PCT_DEFAULT = 0.005
MIN_AREA_PCT_SMALL = 0.001
# Contours smaller than this fraction of the largest contour for the same class are treated as fragments.
CONTOUR_KEEP_RATIO = 0.15

# Vocabulary kept short and generic — retailer search for "crimson" returns junk, "red" works.
COLOR_VOCAB = [
    ("black",    (20, 20, 20)),
    ("white",    (240, 240, 240)),
    ("grey",     (128, 128, 128)),
    ("red",      (200, 40, 40)),
    ("pink",     (240, 150, 180)),
    ("orange",   (240, 140, 50)),
    ("yellow",   (240, 220, 60)),
    ("green",    (60, 160, 70)),
    ("olive",    (120, 130, 60)),
    ("blue",     (50, 90, 200)),
    ("navy",     (30, 40, 90)),
    ("teal",     (40, 140, 150)),
    ("purple",   (130, 60, 180)),
    ("lavender", (200, 180, 230)),
    ("brown",    (110, 70, 40)),
    ("beige",    (220, 200, 170)),
    ("cream",    (245, 235, 215)),
]

CLASS_TO_QUERY_TERM = {
    1: "hat",
    3: "sunglasses",
    4: "top",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    FOOTWEAR_CLASS: "shoes",
    16: "bag",
    17: "scarf",
}


def _rgb_to_color_name(rgb) -> str:
    r, g, b = rgb
    best = None
    best_dist = float("inf")
    for name, (cr, cg, cb) in COLOR_VOCAB:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist = d
            best = name
    return best


def _dominant_color(image_rgb: np.ndarray, mask: np.ndarray):
    """Mean RGB over masked pixels — cheap and stable for solid garments.
    Multi-color patterns give a muddy average; that's acceptable since patterns
    aren't searchable-by-color on retailer keyword search anyway."""
    mask_bool = mask > 0
    if not mask_bool.any():
        return (128, 128, 128)
    pixels = image_rgb[mask_bool]
    mean = pixels.mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _build_query(class_id: int, color_name: str) -> str:
    term = CLASS_TO_QUERY_TERM.get(class_id, "clothing")
    return f"{color_name} {term}"


# --- eBay Browse API adapter ----------------------------------------------
# Docs: https://developer.ebay.com/api-docs/buy/browse/resources/item_summary/methods/search
# Auth: client-credentials OAuth. Token TTL ~7200s, cached in-process.
EBAY_CLIENT_ID = os.environ.get("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.environ.get("EBAY_CLIENT_SECRET")
EBAY_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_CLOTHING_CATEGORY = "11450"  # Clothing, Shoes & Accessories

_ebay_token = {"value": None, "expires_at": 0.0}


def _get_ebay_token() -> str:
    now = time.time()
    if _ebay_token["value"] and now < _ebay_token["expires_at"] - 60:
        return _ebay_token["value"]
    if not (EBAY_CLIENT_ID and EBAY_CLIENT_SECRET):
        raise RuntimeError("EBAY_CLIENT_ID / EBAY_CLIENT_SECRET not configured")
    creds = base64.b64encode(f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        EBAY_TOKEN_URL,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    _ebay_token["value"] = data["access_token"]
    _ebay_token["expires_at"] = now + int(data.get("expires_in", 7200))
    return _ebay_token["value"]


def search_ebay(query: str, limit: int = 10):
    token = _get_ebay_token()
    resp = requests.get(
        EBAY_SEARCH_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "limit": limit, "category_ids": EBAY_CLOTHING_CATEGORY},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    products = []
    for item in data.get("itemSummaries", []):
        price = item.get("price") or {}
        image = item.get("image") or {}
        products.append({
            "title": item.get("title"),
            "image": image.get("imageUrl"),
            "price": price.get("value"),
            "currency": price.get("currency"),
            "url": item.get("itemWebUrl"),
            "source": "ebay",
        })
    return products


def _build_class_mask(seg_map: np.ndarray, class_id: int) -> np.ndarray:
    if class_id == FOOTWEAR_CLASS:
        combined = (seg_map == SHOE_CLASSES[0]) | (seg_map == SHOE_CLASSES[1])
        return combined.astype(np.uint8) * 255
    return (seg_map == class_id).astype(np.uint8) * 255


def _class_confidence(probs: np.ndarray, low_res_seg: np.ndarray, class_id: int) -> float:
    """Mean softmax score over pixels the model argmax-assigned to this class."""
    if class_id == FOOTWEAR_CLASS:
        mask = (low_res_seg == SHOE_CLASSES[0]) | (low_res_seg == SHOE_CLASSES[1])
        if not mask.any():
            return 0.0
        return float((probs[SHOE_CLASSES[0]] + probs[SHOE_CLASSES[1]])[mask].mean())
    mask = (low_res_seg == class_id)
    if not mask.any():
        return 0.0
    return float(probs[class_id][mask].mean())


def _extract_items(class_mask: np.ndarray, class_id: int, width: int, height: int, confidence: float):
    """Clean the mask morphologically, then emit one item per significant contour."""
    # Open removes speckles, close fills pinholes (e.g. skin peeking through a blouse).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    min_area_pct = MIN_AREA_PCT_SMALL if class_id in SMALL_ITEM_CLASSES else MIN_AREA_PCT_DEFAULT
    min_area_px = width * height * min_area_pct
    largest_area = max(cv2.contourArea(c) for c in contours)
    keep_floor = largest_area * CONTOUR_KEEP_RATIO

    items = []
    for idx, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        area = cv2.contourArea(contour)
        if area < min_area_px or area < keep_floor:
            continue

        epsilon = 0.005 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 4:
            continue

        polygon = [[float(pt[0][0]) / width, float(pt[0][1]) / height] for pt in simplified]
        px = [p[0] for p in polygon]
        py = [p[1] for p in polygon]
        bbox = [min(px), min(py), max(px), max(py)]

        items.append({
            "id": f"item_{class_id}_{idx}",
            "class_name": FRIENDLY_NAMES.get(class_id, LABEL_MAP.get(class_id, "Item")),
            "confidence": round(confidence, 3),
            "polygon_normalized": polygon,
            "bbox_normalized": bbox,
            "area_pct": round(area / (width * height), 4),
        })
    return items


class AnalyzeRequest(BaseModel):
    base64_image: str


@app.post("/analyze")
async def analyze_outfit(request: AnalyzeRequest):
    print(f"Received request with base64 image of length: {len(request.base64_image)}")

    try:
        base64_data = request.base64_image
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]

        image_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        print(f"Image opened: {width}x{height}")

        print("Running fashion segmentation...")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = fashion_model(**inputs)

        logits = outputs.logits  # (1, num_classes, h, w) at the model's internal resolution

        # Full-resolution seg_map for polygon extraction.
        upsampled = torch.nn.functional.interpolate(
            logits, size=(height, width), mode='bilinear', align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()

        # Low-res probs for cheap mean-softmax confidence (tall pins produce huge full-res tensors).
        low_res_probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        low_res_seg = logits.argmax(dim=1).squeeze().cpu().numpy()
        print("Segmentation complete.")

        items = []
        for class_id in SHOPPABLE_CLASSES:
            confidence = _class_confidence(low_res_probs, low_res_seg, class_id)
            if confidence == 0.0:
                continue
            class_mask = _build_class_mask(seg_map, class_id)
            items.extend(_extract_items(class_mask, class_id, width, height, confidence))

        print(f"Successfully extracted {len(items)} clothing items.")
        return {
            "status": "success",
            "message": f"Processed image. Found {len(items)} items.",
            "image_size": {"width": width, "height": height},
            "items": items,
        }

    except Exception as e:
        print(f"CRITICAL ERROR processing image: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Backend Error: {str(e)}",
            "items": [],
        }


class MatchRequest(BaseModel):
    base64_image: str
    class_id: int
    polygon_normalized: list  # [[x, y], ...] with x, y in [0, 1]


@app.post("/match")
async def match_item(request: MatchRequest):
    """Given the original image plus one polygon from /analyze, return shoppable products.

    The extension passes back the normalized polygon from /analyze so the backend
    can rasterize it into a mask, pull the dominant color, build a short text
    query (e.g. "lavender top"), and hit retailer search APIs.
    """
    try:
        base64_data = request.base64_image
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]

        image_bytes = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        image_rgb = np.array(img)

        polygon_px = np.array(
            [[int(pt[0] * width), int(pt[1] * height)] for pt in request.polygon_normalized],
            dtype=np.int32,
        )
        if len(polygon_px) < 3:
            return {"status": "error", "message": "Polygon too small", "products": []}

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_px], 255)

        dominant = _dominant_color(image_rgb, mask)
        color_name = _rgb_to_color_name(dominant)
        query = _build_query(request.class_id, color_name)
        print(f"/match query: {query!r} (class_id={request.class_id}, rgb={dominant})")

        try:
            products = search_ebay(query)
        except Exception as search_err:
            print(f"eBay search failed: {search_err}")
            products = []

        return {
            "status": "success",
            "query": query,
            "color": color_name,
            "products": products,
        }

    except Exception as e:
        print(f"CRITICAL ERROR in /match: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e), "products": []}


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
