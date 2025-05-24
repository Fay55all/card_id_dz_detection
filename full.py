from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import easyocr
import pandas as pd
import re
import torch
from datetime import datetime

# Check GPU availability
use_gpu = torch.cuda.is_available()
print(f"🔍 GPU Available: {'Yes ✅' if use_gpu else 'No ❌'}")

# Create output directory
output_dir = "processed_ids"
os.makedirs(output_dir, exist_ok=True)

# Load YOLO models
model_card = YOLO("runs/detect/train25/weights/best.pt")  # ID card detection model
model_flag_front = YOLO("runs/detect/train2/weights/best.pt")  # Front flag detection model
model_flag_back = YOLO("runs/detect/train/weights/best.pt")  # Back flag detection model

# Initialize OCR readers
reader_ar = easyocr.Reader(['ar', 'en'], gpu=use_gpu)  # Arabic/English reader for front side
reader_fr = easyocr.Reader(['fr', 'en'], gpu=use_gpu)  # French/English reader for back side

# ==================== Image Processing Functions ====================

def are_images_similar(img1, img2, threshold=0.7):
    """Compare two images and determine if they are similar"""
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    diff = cv2.absdiff(img1, img2)
    similarity = 1 - (np.sum(diff) / (img1.size * 255))
    return similarity >= threshold, similarity

def extract_card(image_path, output_path):
    """Extract and crop ID card from an image"""
    image = cv2.imread(image_path)
    results = model_card(image_path)

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print("❌ No ID card detected.")
        return None, None

    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped)
    print(f"✅ ID card saved to: {output_path}")
    return cropped, output_path

def extract_flag(image, output_path, model_flag):
    """Extract flag from ID card image"""
    results = model_flag(image)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print("⚠️ No flag found.")
        return None
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    flag = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, flag)
    print(f"✅ Flag saved to: {output_path}")
    return flag

def check_and_fix_orientation(card_img_path, flag_reference_path, model_flag):
    """Check and correct card orientation by comparing flag position"""
    card_img = cv2.imread(card_img_path)
    flag_ref = cv2.imread(flag_reference_path)

    # Extract flag from original image
    flag_output_path = os.path.join(output_dir, f"flag_{os.path.basename(card_img_path)}")
    flag = extract_flag(card_img, flag_output_path, model_flag)
    if flag is None:
        print("⚠️ Cannot verify orientation without flag.")
        return card_img

    # Compare with reference in normal orientation
    match_normal, sim_normal = are_images_similar(flag, flag_ref)

    # Compare rotated flag with reference
    flag_rotated = cv2.rotate(flag, cv2.ROTATE_180)
    match_flipped, sim_flipped = are_images_similar(flag_rotated, flag_ref)

    print(f"🔍 Similarity (normal): {round(sim_normal, 2)}")
    print(f"🔄 Similarity (flipped): {round(sim_flipped, 2)}")

    # Decide if card is in correct orientation
    if sim_normal >= sim_flipped:
        print("✅ Card is in correct orientation.")
        return card_img
    else:
        print("🔁 Incorrect orientation detected, flipping image.")
        flipped_img = cv2.rotate(card_img, cv2.ROTATE_180)
        return flipped_img

def preprocess_image(image_path):
    """Preprocess image for better OCR results"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ OpenCV could not read image: {image_path}")
        return None
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    processed = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9)
    return processed

# ==================== ID Processing Functions ====================

def process_front_image(image_path, flag_reference_path="front_flag_ref.jpg"):
    """Process front side of ID card"""
    timestamp = int(time.time())
    print("\n=== Processing Front Side of ID Card ===")

    # 1. Crop the card
    cropped_path = os.path.join(output_dir, f"front_cropped_{timestamp}.jpg")
    card_img, cropped_path = extract_card(image_path, cropped_path)
    if card_img is None:
        return None

    # 2. Check orientation
    final_card = check_and_fix_orientation(cropped_path, flag_reference_path, model_flag_front)

    # 3. Save final version
    final_output_path = os.path.join(output_dir, f"front_final_{timestamp}.jpg")
    cv2.imwrite(final_output_path, final_card)
    print(f"✅ Final front side saved to: {final_output_path}")
    return final_output_path

def process_back_image(image_path, flag_reference_path="back_flag_ref.jpg"):
    """Process back side of ID card"""
    timestamp = int(time.time())
    print("\n=== Processing Back Side of ID Card ===")

    # 1. Crop the card
    cropped_path = os.path.join(output_dir, f"back_cropped_{timestamp}.jpg")
    card_img, cropped_path = extract_card(image_path, cropped_path)
    if card_img is None:
        return None

    # 2. Check orientation
    final_card = check_and_fix_orientation(cropped_path, flag_reference_path, model_flag_back)

    # 3. Save final version
    final_output_path = os.path.join(output_dir, f"back_final_{timestamp}.jpg")
    cv2.imwrite(final_output_path, final_card)
    print(f"✅ Final back side saved to: {final_output_path}")
    return final_output_path

# ==================== OCR and Data Extraction Functions ====================

def extract_info_front(image_path):
    """Extract information from front side of ID card (Arabic)"""
    image = preprocess_image(image_path)
    if image is None:
        return None

    # Define possible keywords variations due to OCR errors
    keywords = {
        "رقم البطاقة": [r"^\d{9}$"],
        "صلطة الإصدار": ["سلطة الإصدار", "صلطة الإصدار", "ملطة الإصدار", "مللة ا١زصدار", "ملعلة الإسدال"],
        "اللقب": ["اللقب", "تنب", "حلب", "الملب", "اللل", "العنم"],
        "الاسم": ["الاسم", "الإسم", "آلإمم", "الأممم", "الإمىم", "آامم"],
        "الجنس": ["الجنس", "العنس", "لمنس", "آمجنمن", "ججنس", "لعنس", "حلنب "],
        "مكان الميلاد": ["مكان الميلاد", "المبلاد", "مكان حمرلاد", "مكان", "مكان المرا", "المبلاد ", "منان المرك"],
        "تاريخ الميلاد": ["تاريخ الميلاد", "تاربخ الميلاد", "تاربخ لمبلاد", "نار رخ المبلء"],
    }

    results = reader_ar.readtext(image)
    texts = [text[1].strip() for text in results if len(text[1].strip()) > 2]

    print("📄 Extracted text from front side:")
    for t in texts:
        print("-", t)

    info = {
        "رقم البطاقة": "",
        "صلطة الإصدار": "",
        "تاريخ الإصدار": "",
        "تاريخ الانتهاء": "",
        "اللقب": "",
        "الاسم": "",
        "الجنس": "",
        "تاريخ الميلاد": "",
        "مكان الميلاد": ""
    }

    # Extract dates
    dates_found = []
    for text in texts:
        found = re.findall(r"\d{4}[./-]\d{2}[./-]\d{2}", text)
        dates_found.extend(found)

        if re.match(r"^\d{9}$", text):
            info["رقم البطاقة"] = text

        for field, keys in keywords.items():
            for key in keys:
                if key in text:
                    value = text.split(":")[-1].strip()
                    if field == "الجنس":
                        if "ذكر" in text:
                            info[field] = "ذكر"
                        elif "أنثى" in text or "اثثى" in text:
                            info[field] = "أنثى"
                    else:
                        info[field] = value
                    break

    # Sort dates logically, ignoring invalid dates
    valid_dates = []
    for date in dates_found:
        try:
            parsed_date = datetime.strptime(date, "%Y.%m.%d")
            valid_dates.append((parsed_date, date))
        except ValueError:
            print(f"⚠️ Ignoring invalid date: {date}")

    if len(valid_dates) >= 2:
        sorted_dates = sorted(valid_dates, key=lambda x: x[0])
        info["تاريخ الميلاد"] = sorted_dates[0][1]
        if len(sorted_dates) >= 3:
            info["تاريخ الإصدار"] = sorted_dates[1][1]
            info["تاريخ الانتهاء"] = sorted_dates[2][1]
        elif len(sorted_dates) == 2:
            info["تاريخ الإصدار"] = sorted_dates[1][1]
    elif len(valid_dates) == 1:
        info["تاريخ الميلاد"] = valid_dates[0][1]

    if not any(info.values()):
        print("⚠️ No usable data extracted.")
        return None
    return info

def extract_info_back(image_path):
    """Extract information from back side of ID card (French)"""
    image = preprocess_image(image_path)
    if image is None:
        return None

    results = reader_fr.readtext(image)
    texts = [text[1] for text in results]
    print("📄 Extracted text from back side:")
    for t in texts:
        print("-", t)

    info = {
        "Nom": "",
        "Prénom(s)": "",
        "Numéro de série": ""
    }

    # Extract name and surname from MRZ line if reliable
    for text in texts:
        if '<<' in text and not info["Nom"] and not info["Prénom(s)"]:
            parts = text.split("<<")
            if len(parts) >= 2:
                nom = parts[0].replace('<', '').strip()
                prenom = parts[1].replace('<', '').strip()
                if nom.isalpha() and prenom.isalpha():
                    info["Nom"] = nom
                    info["Prénom(s)"] = prenom

    # Look for "Nom" and "Prénom(s)" in texts with the following line
    for i, text in enumerate(texts):
        text_clean = text.strip().upper()

        if "NOM" in text_clean and not info["Nom"]:
            parts = text.split(":")
            if len(parts) > 1 and parts[1].strip():
                info["Nom"] = parts[1].strip()
            elif i + 1 < len(texts):
                next_line = texts[i + 1].strip()
                if next_line and next_line.isalpha():
                    info["Nom"] = next_line

        elif ("PRÉNOM" in text_clean or "PRENOM" in text_clean) and not info["Prénom(s)"]:
            parts = text.split(":")
            if len(parts) > 1 and parts[1].strip():
                info["Prénom(s)"] = parts[1].strip()
            elif i + 1 < len(texts):
                next_line = texts[i + 1].strip()
                if next_line and next_line.isalpha():
                    info["Prénom(s)"] = next_line

        elif text_clean.startswith("IDDZA") and not info["Numéro de série"]:
            match = re.match(r"(IDDZA[0-9A-Z]{9,})", text_clean)
            if match:
                info["Numéro de série"] = match.group(1)

    print("📊 Extracted data from back side:")
    for key, value in info.items():
        print(f"{key}: {value}")

    if not any(info.values()):
        print("⚠️ No information detected.")
        return None

    return info

def save_combined_data(front_info, back_info, output_path="dz_id_combined_info.xlsx"):
    """Save both front and back information to a single Excel file"""
    if front_info is None and back_info is None:
        print("❌ No data to save.")
        return
        
    # Combine data
    combined_data = {}
    if front_info:
        combined_data.update(front_info)
    if back_info:
        combined_data.update(back_info)
    
    # Create DataFrame
    df_new = pd.DataFrame([combined_data])
    
    # Append or create Excel file
    if os.path.exists(output_path):
        df_existing = pd.read_excel(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_excel(output_path, index=False)
    print(f"✅ All data successfully saved to: {output_path}")

# ==================== Main Process Function ====================

def process_id_card(front_image_path, back_image_path, 
                   front_flag_reference="front_flag_ref.jpg", 
                   back_flag_reference="back_flag_ref.jpg",
                   excel_output="dz_id_combined_info.xlsx"):
    """Process both sides of ID card and extract all information"""
    print("\n🔄 Starting ID card processing...\n")

    # Process front side
    front_final_path = process_front_image(front_image_path, front_flag_reference)
    front_info = None
    if front_final_path:
        front_info = extract_info_front(front_final_path)
        if not front_info:
            print("⚠️ Could not extract information from front side.")
    
    # Process back side
    back_final_path = process_back_image(back_image_path, back_flag_reference)
    back_info = None
    if back_final_path:
        back_info = extract_info_back(back_final_path)
        if not back_info:
            print("⚠️ Could not extract information from back side.")
    
    # Save combined data
    if front_info or back_info:
        save_combined_data(front_info, back_info, excel_output)
        print("\n✅ ID card processed successfully!")
        return front_final_path, back_final_path, front_info, back_info
    else:
        print("\n❌ Error during ID card processing.")
        return None, None, None, None

# ==================== Main Execution ====================

if __name__ == "__main__":
    front_path = "2.jpg"           # Front side image path
    back_path = "2b.jpg"          # Back side image path
    front_flag_ref = "front_flag_ref.jpg"  # Reference flag image for front side
    back_flag_ref = "back_flag_ref.jpg"    # Reference flag image for back side
    excel_output = "dz_id_combined_info.xlsx"  # Output Excel file
    
    # Process ID card
    front_result, back_result, front_data, back_data = process_id_card(
        front_path, back_path, front_flag_ref, back_flag_ref, excel_output
    )