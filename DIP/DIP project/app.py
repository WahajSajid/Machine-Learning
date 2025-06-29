from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# Ensure uploads are saved in static/uploads for Flask to serve them
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained tumor detection model
model = load_model('tumor_detector.h5')

def is_probable_mri(img):
    """Basic validation that the image is likely to be a brain MRI."""
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    elif len(img.shape) != 2:
        return False
    if img.shape[0] < 64 or img.shape[1] < 64:
        return False
    img_float = img.astype(np.float32)
    if np.var(img_float) < 50:
        return False
    mean_intensity = np.mean(img_float)
    if mean_intensity < 10 or mean_intensity > 245:
        return False
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < 0.01:
        return False
    return True

def segment_tumor(img, threshold_val=200, dilate_kernel_size=51):
    """
    Segments only the brightest (white) tumor portion in the MRI.
    Returns mask, contours, threshold value, and the thresholded image.
    The mask is dilated to make the detected region thick (dilation after mask construction).
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    min_area = 100
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            mask_temp = np.zeros_like(img)
            cv2.drawContours(mask_temp, [contour], -1, 255, -1)
            mean_val = cv2.mean(img, mask=mask_temp)[0]
            if mean_val > threshold_val:
                cv2.drawContours(mask, [contour], -1, 255, -1)
    # Dilation to thicken the detected region (AFTER mask creation)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    return mask, contours, threshold_val, thresh

def overlay_mask_and_contour_on_image(img, mask, contours):
    """
    Overlay the segmentation mask in red and contours in green on the grayscale MRI image.
    """
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    red_overlay = img_color.copy()
    red_overlay[mask > 0] = [0, 0, 255]
    overlay = cv2.addWeighted(img_color, 0.7, red_overlay, 0.3, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay

def save_threshold_image(thresh_img):
    """
    Save the thresholded image as a PNG for web display.
    Returns the relative file path for Flask static serving.
    """
    thresh_filename = f"thresh_{uuid.uuid4().hex}.png"
    thresh_path = os.path.join(UPLOAD_FOLDER, thresh_filename)
    threechan = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(thresh_path, threechan)
    return f"uploads/{thresh_filename}"

def predict_tumor(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return "Invalid image", None, None
    img_resized = cv2.resize(img, (128, 128))
    if not is_probable_mri(img_resized):
        return "Invalid or not an MRI scanned image of brain", None, None
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, 128, 128, 1)
    prediction = model.predict(img_input)[0][0]
    if prediction > 0.5:
        mask, contours, threshold_val, thresh_img = segment_tumor(img_resized)
        overlay = overlay_mask_and_contour_on_image(img_resized, mask, contours)
        seg_filename = f"seg_{uuid.uuid4().hex}.png"
        seg_path = os.path.join(UPLOAD_FOLDER, seg_filename)
        cv2.imwrite(seg_path, overlay)
        thresh_relpath = save_threshold_image(thresh_img)
        return f"Tumor Detected (Threshold={threshold_val})", f"uploads/{seg_filename}", thresh_relpath
    else:
        return "No Tumor", None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('result.html', result='No file part', seg_path=None, thresh_path=None)
        file = request.files['image']
        if file.filename == '':
            return render_template('result.html', result='No selected file', seg_path=None, thresh_path=None)
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return render_template('result.html', result='Unsupported file format', seg_path=None, thresh_path=None)
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        result, seg_path, thresh_path = predict_tumor(save_path)
        return render_template('result.html', result=result, seg_path=seg_path, thresh_path=thresh_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)