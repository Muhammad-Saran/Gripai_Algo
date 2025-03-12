import cv2
import numpy as np
import mediapipe as mp
import math
import os

# Suppress MediaPipe/TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Constants
REAL_COIN_DIAMETER_MM = 24.26  # US quarter diameter

def get_coin_diameter():
    return REAL_COIN_DIAMETER_MM

def detect_hand_region(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            xs = [int(l.x * image.shape[1]) for l in landmarks.landmark]
            ys = [int(l.y * image.shape[0]) for l in landmarks.landmark]
            # Expand bounding box by 20%
            min_x = max(0, int(min(xs) - 0.2 * (max(xs) - min(xs))))
            max_x = min(image.shape[1], int(max(xs) + 0.2 * (max(xs) - min(xs))))
            min_y = max(0, int(min(ys) - 0.2 * (max(ys) - min(ys))))
            max_y = min(image.shape[0], int(max(ys) + 0.2 * (max(ys) - min(ys))))
            return (min_x, min_y, max_x, max_y)
    return None

def detect_hand(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return hand_landmarks
    return None

def detect_coin(image):
    """
    Detects the coin over the entire image and draws a green circle on it.
    Returns the coin's diameter in pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Dynamic radius estimation based on full image dimensions
    height, width = blurred.shape
    max_dim = max(height, width)
    min_radius = int(max_dim * 0.02)  # tweak these factors as needed
    max_radius = int(max_dim * 0.2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # Fallback to contour analysis if Hough Circle Detection fails
    if circles is None:
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_circle = None
        max_circularity = 0.7
        for cnt in contours:
            if len(cnt) < 5:
                continue
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > max_circularity:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if min_radius <= radius <= max_radius:
                    best_circle = (int(x), int(y), int(radius))
                    max_circularity = circularity
        if best_circle:
            circles = np.array([[[best_circle[0], best_circle[1], best_circle[2]]]])

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            return 2 * r  # return diameter in pixels

    return None

def calculate_scale_factor(coin_pixels, real_diameter=REAL_COIN_DIAMETER_MM):
    if not coin_pixels or coin_pixels <= 0:
        raise ValueError("Invalid coin diameter in pixels")
    return real_diameter / coin_pixels

def get_hand_metrics(landmarks, img_height, img_width):
    def distance(a, b):
        return math.hypot(a.x * img_width - b.x * img_width, a.y * img_height - b.y * img_height)
    
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    return {
        'length': distance(wrist, middle_tip),
        'trigger': distance(wrist, index_tip),
        'grip': distance(thumb_tip, pinky_tip),
        'width': distance(index_mcp, pinky_mcp) * 1.29
    }

def find_category(length_in, width_in):
    length_range = (6.6, 8.6)
    width_range = (2.7, 3.98)
    
    length_norm = max(0, min(100, (length_in - length_range[0]) / (length_range[1] - length_range[0]) * 100))
    width_norm = max(0, min(100, (width_in - width_range[0]) / (width_range[1] - width_range[0]) * 100))
    
    return round((width_norm * 0.6) + (length_norm * 0.4))

def process_hand_scan_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Detect coin over the full image
    coin_diameter_px = detect_coin(image)
    if not coin_diameter_px:
        print("Error: Coin detection failed")
        return

    # Detect hand landmarks and draw them on the image
    hand_landmarks = detect_hand(image)
    if not hand_landmarks:
        print("Error: Hand detection failed")
        return

    # Calculate metrics and scale
    scale = calculate_scale_factor(coin_diameter_px)
    metrics = get_hand_metrics(hand_landmarks, *image.shape[:2])
    
    # Convert to real-world measurements (mm) and then inches
    hand_length_mm = metrics['length'] * scale
    hand_width_mm = metrics['width'] * scale
    length_in = hand_length_mm / 25.4
    width_in = hand_width_mm / 25.4

    # Determine the weighted category
    category = find_category(length_in, width_in)

    # Print the results
    print(f"Hand Length: {length_in:.2f} inches")
    print(f"Hand Width: {width_in:.2f} inches")
    print(f"Weighted Category: {category}")

    # Save the output image with coin circle and hand landmarks
    os.makedirs("processed_images", exist_ok=True)
    output_path = os.path.join("processed_images", os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Saved processed image to: {output_path}")

def main():
    image_path = input("Enter image path: ").strip()
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return
    
    print("\nProcessing...")
    process_hand_scan_image(image_path)

if __name__ == "__main__":
    main()
