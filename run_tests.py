import os
import requests
from PIL import Image, ImageDraw

def create_test_images():
    print("Generating 5 different test images...")
    os.makedirs("test_images", exist_ok=True)
    
    # 5 denominations to generate test images for
    test_denoms = [
        ("1000_Taka", (220, 160, 175)),
        ("500_Taka", (160, 190, 175)),
        ("100_Taka", (160, 160, 220)),
        ("50_Taka", (180, 220, 235)),
        ("20_Taka", (160, 220, 180))
    ]
    
    for i, (name, color) in enumerate(test_denoms, 1):
        img_path = f"test_images/test{i}.jpg"
        # Create a test image resembling our training data
        img = Image.new("RGB", (640, 640), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Draw banknote shape
        draw.rectangle([120, 200, 520, 400], fill=color)
        draw.rectangle([126, 206, 514, 394], outline=tuple(max(0, c-40) for c in color), width=2)
        # Draw value text
        draw.text((150, 280), f"{name.replace('_', ' ')}", fill=(0, 0, 0))
        img.save(img_path)
        print(f"  Saved test image {i}: {img_path} ({name})")

def test_api():
    base_url = "http://localhost:8000"
    
    print("\n--- Testing GET / (Root Endpoint) ---")
    try:
        r = requests.get(f"{base_url}/")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server at http://localhost:8000. Is it running?")
        return

    print("\n--- Testing POST /predict (Valid Request: Test Image 1) ---")
    img_path = "test_images/test1.jpg"
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        r = requests.post(f"{base_url}/predict", files=files)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")

    print("\n--- Testing POST /predict (Valid Request: Test Image 2) ---")
    img_path = "test_images/test2.jpg"
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        r = requests.post(f"{base_url}/predict", files=files)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")

    print("\n--- Testing POST /predict (Valid Request: Test Image 3) ---")
    img_path = "test_images/test3.jpg"
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        r = requests.post(f"{base_url}/predict", files=files)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")

    print("\n--- Testing POST /predict (Valid Request: Test Image 4) ---")
    img_path = "test_images/test4.jpg"
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        r = requests.post(f"{base_url}/predict", files=files)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")

    print("\n--- Testing POST /predict (Valid Request: Test Image 5) ---")
    img_path = "test_images/test5.jpg"
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        r = requests.post(f"{base_url}/predict", files=files)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")

    print("\n--- Testing Error Handling (Invalid File Type) ---")
    temp_txt_path = "test_images/invalid.txt"
    with open(temp_txt_path, "w") as f:
        f.write("This is not an image.")
    
    with open(temp_txt_path, "rb") as f:
        files = {"file": ("invalid.txt", f, "text/plain")}
        r = requests.post(f"{base_url}/predict", files=files)
        print(f"Status Code: {r.status_code} (Expected: 400)")
        print(f"Response: {r.json()}")
    
    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)

    print("\n--- Testing Error Handling (Empty File) ---")
    temp_empty_path = "test_images/empty.jpg"
    with open(temp_empty_path, "wb") as f:
        pass
    
    with open(temp_empty_path, "rb") as f:
        files = {"file": ("empty.jpg", f, "image/jpeg")}
        r = requests.post(f"{base_url}/predict", files=files)
        print(f"Status Code: {r.status_code} (Expected: 400)")
        print(f"Response: {r.json()}")

    if os.path.exists(temp_empty_path):
        os.remove(temp_empty_path)

if __name__ == "__main__":
    create_test_images()
    # API testing can be run by calling this script directly after launching the server.
    test_api()
