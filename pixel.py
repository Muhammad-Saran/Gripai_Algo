import os
import cv2
import numpy as np
import mediapipe as mp
import math
from PIL import Image
from main import HandScannerBase, mp_hands

# Suppress TensorFlow/MediaPipe logging (reinforced here for consistency)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PixelHandScanner(HandScannerBase):
    def __init__(self, image_path):
        super().__init__(image_path)

    @staticmethod
    def calculate_scale_factor(coin_diameter_pixels, real_coin_diameter_mm):
        """Calculate scale factor from coin diameter."""
        real_coin_diameter_mm = float(real_coin_diameter_mm)
        if coin_diameter_pixels == 0 or coin_diameter_pixels is None:
            raise ValueError("Coin diameter in pixels cannot be zero or None")
        return real_coin_diameter_mm / coin_diameter_pixels

    def get_hand_metrics(self, hand_landmarks):
        """Calculate hand metrics and draw a line from middle finger tip to wrist."""
        # Get middle finger tip and wrist landmarks
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Convert landmarks to pixel coordinates
        middle_finger_x = int(middle_finger_tip.x * self.image_width)
        middle_finger_y = int(middle_finger_tip.y * self.image_height)
        wrist_x = int(wrist.x * self.image_width)
        wrist_y = int(wrist.y * self.image_height)

        # Draw the line on the image
        cv2.line(self.image_no_bg, (middle_finger_x, middle_finger_y), (wrist_x, wrist_y), (0, 255, 0), 2)

        # Calculate hand length (distance between middle finger tip and wrist)
        hand_length_pixels = self.get_distance_in_pixels(middle_finger_tip, wrist, self.image_width, self.image_height)

        # Calculate hand width (distance between index finger MCP and pinky MCP)
        hand_width_pixels = self.get_distance_in_pixels(
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
            self.image_width, self.image_height
        )
        adjusted_hand_width_pixels = hand_width_pixels * 1.3

        return hand_length_pixels, adjusted_hand_width_pixels

    def process(self):
        """Process the hand scan image using pixel-based scaling."""
        result = self.detect_coin()
        if result is None:
            print("Error: Could not detect coin in the image")
            return
        coin_diameter_pixels, _, _ = result
        scale_factor = self.calculate_scale_factor(coin_diameter_pixels, self.get_coin_diameter())
        
        hand_landmarks = self.detect_hand()
        if hand_landmarks is None:
            print("Error: Could not detect hand in the image")
            return
        
        hand_length_pixels, hand_width_pixels = self.get_hand_metrics(hand_landmarks)
        hand_length_mm = hand_length_pixels * scale_factor
        hand_width_mm = hand_width_pixels * scale_factor
        hand_length_inches = hand_length_mm / 25.4
        hand_width_inches = hand_width_mm / 25.4
        
        weighted_category = self.find_category(length=hand_length_inches, width=hand_width_inches)
        print(f"Hand Length: {hand_length_inches:.2f} inches")
        print(f"Hand Width: {hand_width_inches:.2f} inches")
        print(f"Weighted Category: {weighted_category}")
        
        self.save_processed_image(hand_length_inches, hand_width_inches)

if __name__ == "__main__":
    image_path = "D:\\Projects\\GripAI_Algo\\Pictures\\Usama\\usama3.png"
    if not os.path.exists(image_path):
        print("Error: Image file does not exist!")
    else:
        print("Processing your hand scan (Pixel-based)...")
        scanner = PixelHandScanner(image_path)
        scanner.process()