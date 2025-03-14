import cv2
import numpy as np
import mediapipe as mp
import math
from PIL import Image
import os
import logging

# Suppress TensorFlow/MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info+, 2 = warning+, 3 = error+
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

class HandScannerBase:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.image_no_bg = None
        self.image_height = None
        self.image_width = None
        self.output_folder = None
        self.base_name = None
        self._load_image()

    def _load_image(self):
        """Load and preprocess the image."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Error: Could not load image from {self.image_path}")
        
        image_pil = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.image_no_bg = image_pil  # Placeholder for background removal
        self.image_no_bg = np.array(self.image_no_bg)
        self.image_no_bg = cv2.cvtColor(self.image_no_bg, cv2.COLOR_RGB2BGR)
        
        self.image_height, self.image_width, _ = self.image_no_bg.shape
        
        base_dir = os.path.dirname(self.image_path)
        self.base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        self.output_folder = os.path.join(base_dir, "processed_images")
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def get_coin_diameter():
        """Return the real-world diameter of a US quarter in mm."""
        return 25.50  # US quarter diameter in mm

    @staticmethod
    def find_category(length, width):
        """Calculate weighted category based on hand length and width."""
        minRangeWidth = 2.7
        maxRangeWidth = 3.98
        minRangeLength = 6.6
        maxRangeLength = 8.6
        positionRangeWidth = ((width - minRangeWidth) / (maxRangeWidth - minRangeWidth)) * 100
        positionRangeLength = ((length - minRangeLength) / (maxRangeLength - minRangeLength)) * 100
        weightedAverage = (positionRangeWidth * 0.6) + (positionRangeLength * 0.4)
        return round(weightedAverage)

    def detect_hand_region(self):
        """Detect the hand region using MediaPipe landmarks."""
        image_rgb = cv2.cvtColor(self.image_no_bg, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                wrist = landmarks.landmark[0]
                index_tip = landmarks.landmark[8]
                pinky_tip = landmarks.landmark[16]
                wrist_x, wrist_y = int(wrist.x * self.image_width), int(wrist.y * self.image_height)
                index_tip_x, index_tip_y = int(index_tip.x * self.image_width), int(index_tip.y * self.image_height)
                pinky_tip_x, pinky_tip_y = int(pinky_tip.x * self.image_width), int(pinky_tip.y * self.image_height)
                min_x = min(wrist_x, index_tip_x, pinky_tip_x)
                min_y = min(wrist_y, index_tip_y, pinky_tip_y)
                max_x = max(wrist_x, index_tip_x, pinky_tip_x)
                max_y = max(wrist_y, index_tip_y, pinky_tip_y)
                return (min_x, min_y, max_x, max_y)
        return None

    def detect_hand(self):
        """Detect and draw hand landmarks using MediaPipe."""
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            image_rgb = cv2.cvtColor(self.image_no_bg, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(self.image_no_bg, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    return hand_landmarks
        return None

    @staticmethod
    def detect_hough_circle_transform(dp, mindist, param1, param2, minradius, maxradius, image):
        """Apply Hough Circle Transform to detect circles (unused)."""
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=dp, minDist=mindist, param1=param1,
            param2=param2, minRadius=minradius, maxRadius=maxradius
        )
        return circles

    def detect_coin(self):
        """Detect red circle (coin) and annotate the image."""
        hand_region = self.detect_hand_region()
        if not hand_region:
            print("Error: No hand region detected.")
            return None

        min_x, min_y, max_x, max_y = hand_region
        cropped_hand = self.image_no_bg[min_y:max_y, min_x:max_x]
        hsv = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        debug_base = os.path.join(self.output_folder, f"debug_{self.base_name}")
        debug_path = f"{debug_base}.jpg"
        counter = 2
        while os.path.exists(debug_path):
            debug_path = f"{debug_base}_{counter}.jpg"
            counter += 1
        cv2.imwrite(debug_path, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                ellipse = cv2.fitEllipse(largest_contour)
                center, axes, angle = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                diameter_pixels = int(minor_axis)
                x, y = int(center[0]), int(center[1])
                x_full, y_full = x + min_x, y + min_y
                cv2.ellipse(self.image_no_bg, (x_full, y_full), (int(axes[0]/2), int(axes[1]/2)), angle, 0, 360, (0, 255, 0), 3)
                
                if major_axis == axes[0]:
                    start_point = (int(x_full - axes[1]/2 * math.cos(math.radians(angle + 90))), 
                                   int(y_full - axes[1]/2 * math.sin(math.radians(angle + 90))))
                    end_point = (int(x_full + axes[1]/2 * math.cos(math.radians(angle + 90))), 
                                 int(y_full + axes[1]/2 * math.sin(math.radians(angle + 90))))
                else:
                    start_point = (int(x_full - axes[0]/2 * math.cos(math.radians(angle))), 
                                   int(y_full - axes[0]/2 * math.sin(math.radians(angle))))
                    end_point = (int(x_full + axes[0]/2 * math.cos(math.radians(angle))), 
                                 int(y_full + axes[0]/2 * math.sin(math.radians(angle))))
                cv2.line(self.image_no_bg, start_point, end_point, (255, 0, 0), 2)
                cv2.putText(self.image_no_bg, f'Diameter: {diameter_pixels}px',
                            (x_full - int(axes[0]/2), y_full - int(axes[1]/2) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print(f"Detected diameter: {diameter_pixels} pixels")
                return diameter_pixels, start_point, end_point
        print("Error: No red circle detected.")
        return None

    @staticmethod
    def get_distance_in_pixels(landmark1, landmark2, image_width, image_height):
        """Calculate Euclidean distance between two landmarks in pixels."""
        x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
        x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def save_processed_image(self, hand_length_inches, hand_width_inches):
        """Save the processed image with annotations."""
        text = f"Length: {hand_length_inches:.2f} in, Width: {hand_width_inches:.2f} in"
        cv2.putText(self.image_no_bg, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        output_base = os.path.join(self.output_folder, f"{self.base_name}_processed")
        output_path = f"{output_base}.jpg"
        counter = 2
        while os.path.exists(output_path):
            output_path = f"{output_base}{counter}.jpg"
            counter += 1
        cv2.imwrite(output_path, self.image_no_bg)
        print(f"Processed image saved as: {output_path}")