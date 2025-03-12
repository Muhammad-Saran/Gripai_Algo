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

def detect_hough_circle_transform(image, dp=1.2, min_dist=50, param1=100, param2=30, min_radius=30, max_radius=200):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    return circles

def detect_coin(image):
    hand_region = detect_hand_region(image)

    if hand_region:
        min_x, min_y, max_x, max_y = hand_region
        cropped_hand = image[min_y:max_y, min_x:max_x]

        # Preprocess the image
        gray = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (21, 21), 2)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Detect circles using Hough Circle Transform
        circles = detect_hough_circle_transform(edges)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                x_full, y_full = x + min_x, y + min_y
                cv2.circle(image, (x_full, y_full), r, (0, 255, 0), 4)
                diameter_pixels = 2 * r
                cv2.putText(image, f'Diameter: {diameter_pixels}px',
                            (x_full - r, y_full - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                return diameter_pixels

        # Fallback to contour-based detection if Hough Circle fails
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 5:  # Minimum points for ellipse fitting
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)

                # Filter based on aspect ratio (to exclude non-circular shapes)
                if 0.8 <= (minor_axis / major_axis) <= 1.2:
                    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                    diameter_pixels = (major_axis + minor_axis) / 2
                    x_full, y_full = int(center[0]) + min_x, int(center[1]) + min_y
                    cv2.putText(image, f'Diameter: {diameter_pixels:.2f}px',
                                (x_full - 50, y_full - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    return diameter_pixels

    print("Error: No coin detected.")
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

    coin_diameter_pixels = detect_coin(image_no_bg)
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

    output_folder = "processed_images"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "processed_hand_image_with_coin_and_landmarks.jpg")
    cv2.imwrite(output_path, image_no_bg)
    print(f"Processed image with coin detection and hand landmarks saved as: {output_path}")

def main():
    image_path = input("Please enter the path to your hand scan image: ")
    if not os.path.exists(image_path):
        print("Error: Image file does not exist!")
        return
    print("Processing your hand scan...")
    process_hand_scan_image(image_path)

if __name__ == "__main__":
    main()
    
    
#complete
    
    