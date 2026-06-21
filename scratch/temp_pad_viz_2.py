import os
import cv2
import numpy as np
from pathlib import Path

def find_non_square_image():
    # Candidates directories
    paths = [
        Path("d:/Xu Ly Anh So/ProjectNhom/vehicle-type-recognition/data"),
        Path("d:/Xu Ly Anh So/ProjectNhom/vehicle-type-recognition"),
        Path("d:/Xu Ly Anh So/ProjectNhom"),
        Path("d:/Xu Ly Anh So")
    ]
    
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Recursively search for a non-square image
    for p in paths:
        if p.exists():
            for root, dirs, files in os.walk(p):
                # Avoid venv and git directories
                if "venv" in root or ".git" in root or "scratch" in root:
                    continue
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in img_exts:
                        img_path = Path(root) / f
                        if "padding_comparison" not in f and "~$" not in f:
                            # Verify aspect ratio is non-square
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                h, w = img.shape[:2]
                                # Check if it is significantly non-square (difference of at least 40 pixels)
                                if abs(w - h) > 40:
                                    return img_path, img
    return None, None

def main():
    img_path, img = find_non_square_image()
    if not img_path or img is None:
        print("No non-square image found!")
        return
        
    print(f"Using non-square image: {img_path} (original size: {img.shape})")
    
    # Resize keeping aspect ratio (longest edge = 224)
    image_size = 224
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding sizes
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 1. Zero Padding
    zero_padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # 2. Reflective Padding (BORDER_REFLECT_101)
    reflect_padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    
    # Create canvas
    spacing = 15
    canvas_w = 224 * 3 + spacing * 2
    canvas_h = 224 + 40
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255 # White background
    
    # Center original resized image in the first box
    orig_box = np.ones((224, 224, 3), dtype=np.uint8) * 240 # Light grey box
    orig_x = (224 - new_w) // 2
    orig_y = (224 - new_h) // 2
    orig_box[orig_y:orig_y+new_h, orig_x:orig_x+new_w] = resized
    
    canvas[0:224, 0:224] = orig_box
    canvas[0:224, 224 + spacing : 224*2 + spacing] = zero_padded
    canvas[0:224, 224*2 + spacing*2 : 224*3 + spacing*2] = reflect_padded
    
    # Draw dark grey borders around the three boxes
    for offset in [0, 224 + spacing, 224*2 + spacing*2]:
        cv2.rectangle(canvas, (offset, 0), (offset + 224, 224), (180, 180, 180), 1)
        
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original (Resized)", (10, 245), font, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Zero Padding (V1/V2)", (224 + spacing + 10, 245), font, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Reflective Padding (V3)", (224*2 + spacing*2 + 10, 245), font, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    
    # Save the output image
    output_dir = Path("d:/Xu Ly Anh So/ProjectNhom/vehicle-type-recognition/docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "padding_comparison.png"
    cv2.imwrite(str(output_path), canvas)
    print(f"Comparison image successfully saved to: {output_path}")

if __name__ == '__main__':
    main()
