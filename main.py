import cv2
import numpy as np
import mediapipe as mp
import math
from PIL import Image
import os

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Simulated functions
def get_coin_diameter():
    return 24.26  # US quarter diameter in mm

def find_category(length, width):
    minRangeWidth = 2.7
    maxRangeWidth = 3.98
    minRangeLength = 6.6
    maxRangeLength = 8.6

    positionRangeWidth = ((width - minRangeWidth) / (maxRangeWidth - minRangeWidth)) * 100
    positionRangeLength = ((length - minRangeLength) / (maxRangeLength - minRangeLength)) * 100

    weightedAverage = (positionRangeWidth * 0.6) + (positionRangeLength * 0.4)
    weight = round(weightedAverage)
    return weight

def detect_hand_region(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            wrist = landmarks.landmark[0]
            index_tip = landmarks.landmark[8]
            pinky_tip = landmarks.landmark[16]

            h, w, _ = image.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)
            pinky_tip_x, pinky_tip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)

            min_x = min(wrist_x, index_tip_x, pinky_tip_x)
            min_y = min(wrist_y, index_tip_y, pinky_tip_y)
            max_x = max(wrist_x, index_tip_x, pinky_tip_x)
            max_y = max(wrist_y, index_tip_y, pinky_tip_y)

            return (min_x, min_y, max_x, max_y)
    return None

def detect_hand(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                return hand_landmarks
    return None

def detect_hough_circle_transform(dp, mindist, param1, param2, minradius, maxradius, image):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=mindist,
        param1=param1,
        param2=param2,
        minRadius=minradius,
        maxRadius=maxradius
    )
    return circles

def detect_coin(image, base_dir, base_name):
    hand_region = detect_hand_region(image)
    if not hand_region:
        print("Error: No hand region detected.")
        return None

    min_x, min_y, max_x, max_y = hand_region
    cropped_hand = image[min_y:max_y, min_x:max_x]
    gray = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # First attempt: Sobel edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_magnitude = np.uint8(255 * edge_magnitude / np.max(edge_magnitude))
    _, edges_sobel = cv2.threshold(edge_magnitude, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_sobel = cv2.morphologyEx(edges_sobel, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(os.path.join(base_dir, f"debug_{base_name}_sobel.jpg"), edges_sobel)  # Save in Mueen dir

    circles = detect_hough_circle_transform(1.2, 50, 80, 20, 10, 50, edges_sobel)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            x_full, y_full = x + min_x, y + min_y
            cv2.circle(image, (x_full, y_full), r, (0, 255, 0), 4)
            start_point = (x_full - r, y_full)
            end_point = (x_full + r, y_full)
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)
            diameter_pixels = 2 * r
            cv2.putText(image, f'Diameter: {diameter_pixels}px',
                        (x_full - r, y_full - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            return diameter_pixels

    # Second attempt: Canny-based fallback
    print("Sobel attempt failed, trying Canny-based detection...")
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, threshold1=20, threshold2=80)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(os.path.join(base_dir, f"debug_{base_name}_canny.jpg"), edges)  # Save in Mueen dir

    circles = detect_hough_circle_transform(1.2, 50, 80, 20, 10, 50, edges)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            x_full, y_full = x + min_x, y + min_y
            cv2.circle(image, (x_full, y_full), r, (0, 255, 0), 4)
            start_point = (x_full - r, y_full)
            end_point = (x_full + r, y_full)
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)
            diameter_pixels = 2 * r
            cv2.putText(image, f'Diameter: {diameter_pixels}px',
                        (x_full - r, y_full - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            return diameter_pixels

    print("Error: No circles detected after multiple attempts.")
    return None

def calculate_scale_factor(coin_diameter_pixels, real_coin_diameter_mm):
    real_coin_diameter_mm = float(real_coin_diameter_mm)
    if coin_diameter_pixels == 0 or coin_diameter_pixels is None:
        raise ValueError("Coin diameter in pixels cannot be zero or None")
    return real_coin_diameter_mm / coin_diameter_pixels

def get_distance_in_pixels(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
    x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_hand_metrics(hand_landmarks, image_height, image_width):
    hand_length_pixels = get_distance_in_pixels(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                                                image_width, image_height)
    trigger_distance_pixels = get_distance_in_pixels(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                                                     hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                                                     image_width, image_height)
    grip_length_pixels = get_distance_in_pixels(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                                                image_width, image_height)
    hand_width_pixels = get_distance_in_pixels(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                                               hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                                               image_width, image_height)
    adjusted_hand_width_pixels = hand_width_pixels * 1.29
    return hand_length_pixels, trigger_distance_pixels, grip_length_pixels, adjusted_hand_width_pixels

def process_hand_scan_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image from", image_path)
        return

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_no_bg = image_pil  # Uncomment and install rembg if needed: image_no_bg = remove(image_pil)

    image_no_bg = np.array(image_no_bg)
    image_no_bg = cv2.cvtColor(image_no_bg, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image_no_bg.shape

    # Extract directory and filename from image_path
    base_dir = os.path.dirname(image_path)  # e.g., D:\Projects\GripAI_Algo\Pictures\Mueen
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # e.g., mueen1

    coin_diameter_pixels = detect_coin(image_no_bg, base_dir, base_name)
    if coin_diameter_pixels is None:
        print("Error: Could not detect coin in the image")
        return

    scale_factor = calculate_scale_factor(coin_diameter_pixels, get_coin_diameter())

    hand_landmarks = detect_hand(image_no_bg)
    if hand_landmarks is None:
        print("Error: Could not detect hand in the image")
        return

    hand_length_pixels, trigger_distance_pixels, grip_length_pixels, hand_width_pixels = get_hand_metrics(
        hand_landmarks, image_height, image_width
    )

    hand_length_mm = hand_length_pixels * scale_factor
    hand_width_mm = hand_width_pixels * scale_factor

    hand_length_inches = hand_length_mm / 25.4
    hand_width_inches = hand_width_mm / 25.4

    weighted_category = find_category(length=hand_length_inches, width=hand_width_inches)

    print(f"Hand Length: {hand_length_inches:.2f} inches")
    print(f"Hand Width: {hand_width_inches:.2f} inches")
    print(f"Weighted Category: {weighted_category}")

    # Create processed_images folder inside the base directory (e.g., Mueen)
    output_folder = os.path.join(base_dir, "processed_images")
    os.makedirs(output_folder, exist_ok=True)  # Reuse if exists

    # Add length and width text to the top-left corner of the image
    text = f"Length: {hand_length_inches:.2f} in, Width: {hand_width_inches:.2f} in"
    cv2.putText(image_no_bg, text, (10, 30),  # Top-left corner (x=10, y=30)
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save processed image with new naming convention
    output_path = os.path.join(output_folder, f"{base_name}_processed.jpg")
    cv2.imwrite(output_path, image_no_bg)
    print(f"Processed image saved as: {output_path}")

def main():
    image_path = input("Please enter the path to your hand scan image: ")
    if not os.path.exists(image_path):
        print("Error: Image file does not exist!")
        return
    print("Processing your hand scan...")
    process_hand_scan_image(image_path)

if __name__ == "__main__":
    main()