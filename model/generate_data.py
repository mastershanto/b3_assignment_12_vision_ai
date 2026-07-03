import os
import random
import math
from PIL import Image, ImageDraw, ImageFont

# Define classes and their representative colors (RGB)
DENOMINATIONS = {
    "2_Taka": (245, 200, 180),    # Peach
    "5_Taka": (190, 210, 195),    # Light Greenish-Gray
    "10_Taka": (240, 185, 210),   # Pink/Lavender
    "20_Taka": (160, 220, 180),   # Light Green
    "50_Taka": (180, 220, 235),   # Light Blue/Cyan
    "100_Taka": (160, 160, 220),  # Purple/Blue
    "500_Taka": (160, 190, 175),  # Green-Gray
    "1000_Taka": (220, 160, 175)  # Red-Violet
}

CLASSES = list(DENOMINATIONS.keys())

def create_synthetic_image(output_img_path, output_lbl_path, image_size=640):
    # 1. Create a random background canvas
    bg_mode = random.choice(["solid", "noise", "shapes"])
    if bg_mode == "solid":
        bg_color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
        img = Image.new("RGB", (image_size, image_size), bg_color)
    elif bg_mode == "noise":
        img = Image.new("RGB", (image_size, image_size))
        pixels = img.load()
        base_color = [random.randint(180, 220) for _ in range(3)]
        for y in range(image_size):
            for x in range(image_size):
                noise = random.randint(-15, 15)
                pixels[x, y] = tuple(max(0, min(255, c + noise)) for c in base_color)
    else: # shapes
        bg_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
        img = Image.new("RGB", (image_size, image_size), bg_color)
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(5, 15)):
            x1, y1 = random.randint(0, image_size), random.randint(0, image_size)
            x2, y2 = x1 + random.randint(20, 100), y1 + random.randint(20, 100)
            shape_color = (random.randint(180, 210), random.randint(180, 210), random.randint(180, 210))
            draw.rectangle([x1, y1, x2, y2], fill=shape_color)

    # 2. Choose a random denomination
    denom = random.choice(CLASSES)
    class_idx = CLASSES.index(denom)
    denom_color = DENOMINATIONS[denom]

    # 3. Create banknote image with alpha channel for transparency rotation
    note_w = random.randint(280, 420)
    note_h = random.randint(120, 180)
    note_img = Image.new("RGBA", (note_w, note_h), (0, 0, 0, 0))
    note_draw = ImageDraw.Draw(note_img)

    # Draw the banknote body
    note_draw.rectangle([0, 0, note_w, note_h], fill=denom_color + (255,))
    
    # Draw simple design borders
    border_offset = 6
    border_color = tuple(max(0, c - 40) for c in denom_color) + (255,)
    note_draw.rectangle([border_offset, border_offset, note_w - border_offset, note_h - border_offset], outline=border_color, width=2)
    
    # Draw a security thread
    thread_x = int(note_w * 0.75)
    note_draw.line([thread_x, border_offset, thread_x, note_h - border_offset], fill=(180, 180, 180, 255), width=3)

    # Draw denomination text
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Overlay value and denomination labels
    val_str = denom.split("_")[0]
    note_draw.text((20, 20), val_str, fill=(0, 0, 0, 255), font=font)
    note_draw.text((20, 40), "BANGLADESH BANK", fill=border_color, font=font)
    note_draw.text((note_w // 2 - 30, note_h // 2 - 10), denom.replace("_", " "), fill=(0, 0, 0, 255), font=font)
    note_draw.text((note_w - 60, note_h - 30), "TAKA", fill=(0, 0, 0, 255), font=font)

    # 4. Rotate the banknote
    angle = random.randint(-45, 45)
    # expand=True changes output size to fit rotated content
    rotated_note = note_img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # 5. Place rotated banknote onto background
    # Choose placement offset
    max_x = image_size - rotated_note.width
    max_y = image_size - rotated_note.height
    
    if max_x <= 0 or max_y <= 0:
        # Fallback to center if note is too big
        offset_x = max(0, (image_size - rotated_note.width) // 2)
        offset_y = max(0, (image_size - rotated_note.height) // 2)
    else:
        offset_x = random.randint(0, max_x)
        offset_y = random.randint(0, max_y)

    img.paste(rotated_note, (offset_x, offset_y), rotated_note)

    # 6. Calculate bounding box of the pasted rotated note (alpha channel > 0)
    alpha = rotated_note.split()[3]
    bbox_local = alpha.getbbox() # (left, upper, right, lower) of non-transparent part
    
    if bbox_local:
        xmin = offset_x + bbox_local[0]
        ymin = offset_y + bbox_local[1]
        xmax = offset_x + bbox_local[2]
        ymax = offset_y + bbox_local[3]
    else:
        # Fallback if bbox calculation fails
        xmin = offset_x
        ymin = offset_y
        xmax = offset_x + rotated_note.width
        ymax = offset_y + rotated_note.height

    # Clamping
    xmin = max(0, min(xmin, image_size - 1))
    ymin = max(0, min(ymin, image_size - 1))
    xmax = max(xmin + 1, min(xmax, image_size))
    ymax = max(ymin + 1, min(ymax, image_size))

    # Convert to YOLO format
    x_center = (xmin + xmax) / 2.0 / image_size
    y_center = (ymin + ymax) / 2.0 / image_size
    w = (xmax - xmin) / image_size
    h = (ymax - ymin) / image_size

    # Save image
    img.save(output_img_path)

    # Save label
    with open(output_lbl_path, "w") as f:
        f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def generate_dataset(num_train=80, num_val=20):
    print("Generating synthetic currency dataset...")
    base_dir = "dataset"
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "val"), exist_ok=True)

    # Generate Train Set
    for i in range(num_train):
        img_path = os.path.join(base_dir, "images", "train", f"train_{i}.jpg")
        lbl_path = os.path.join(base_dir, "labels", "train", f"train_{i}.txt")
        create_synthetic_image(img_path, lbl_path)

    # Generate Validation Set
    for i in range(num_val):
        img_path = os.path.join(base_dir, "images", "val", f"val_{i}.jpg")
        lbl_path = os.path.join(base_dir, "labels", "val", f"val_{i}.txt")
        create_synthetic_image(img_path, lbl_path)

    # Write YOLO dataset.yaml config
    abs_base_dir = os.path.abspath(base_dir)
    yaml_content = f"""path: {abs_base_dir}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(CLASSES):
        yaml_content += f"  {idx}: {name}\n"

    with open("dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Dataset generation complete! Generated {num_train} train and {num_val} val images.")
    print("dataset.yaml configuration file written.")

if __name__ == "__main__":
    generate_dataset()
