import os
import cv2
import numpy as np
import mediapipe as mp
import math
from PIL import Image
from main import HandScannerBase, mp_hands

# Suppress TensorFlow/MediaPipe logging (reinforced here for consistency)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DiameterHandScanner(HandScannerBase):
    def __init__(self, image_path):
        super().__init__(image_path)

    @staticmethod
    def calculate_line_length(start_point, end_point):
        """Calculate the Euclidean distance between two points."""
        x1, y1 = start_point
        x2, y2 = end_point
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_hand_metrics(self, hand_landmarks, diameter_line_length):
        """Calculate hand metrics using the coin's diameter line length."""
        hand_length_pixels = self.get_distance_in_pixels(
            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            self.image_width, self.image_height
        )
        trigger_distance_pixels = self.get_distance_in_pixels(
            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
            self.image_width, self.image_height
        )
        grip_length_pixels = self.get_distance_in_pixels(
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
            self.image_width, self.image_height
        )
        hand_width_pixels = self.get_distance_in_pixels(
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
            self.image_width, self.image_height
        )
        mm_per_pixel = self.get_coin_diameter() / diameter_line_length
        hand_length_mm = hand_length_pixels * mm_per_pixel
        trigger_distance_mm = trigger_distance_pixels * mm_per_pixel
        grip_length_mm = grip_length_pixels * mm_per_pixel
        hand_width_mm = hand_width_pixels * mm_per_pixel
        hand_length_inches = hand_length_mm / 25.4
        trigger_distance_inches = trigger_distance_mm / 25.4
        grip_length_inches = grip_length_mm / 25.4
        hand_width_inches = hand_width_mm / 25.4
        return hand_length_inches, trigger_distance_inches, grip_length_inches, hand_width_inches

    def process(self):
        """Process the hand scan image using diameter-based scaling."""
        result = self.detect_coin()
        if result is None:
            print("Error: Could not detect coin in the image")
            return
        diameter_pixels, start_point, end_point = result
        diameter_line_length = self.calculate_line_length(start_point, end_point)
        print(f"Diameter line length: {diameter_line_length:.2f} pixels")
        
        hand_landmarks = self.detect_hand()
        if hand_landmarks is None:
            print("Error: Could not detect hand in the image")
            return
        
        hand_length_inches, _, _, hand_width_inches = self.get_hand_metrics(hand_landmarks, diameter_line_length)
        weighted_category = self.find_category(length=hand_length_inches, width=hand_width_inches)
        print(f"Hand Length: {hand_length_inches:.2f} inches")
        print(f"Hand Width: {hand_width_inches:.2f} inches")
        print(f"Weighted Category: {weighted_category}")
        
        self.save_processed_image(hand_length_inches, hand_width_inches)

if __name__ == "__main__":
    image_path = "D:\\Projects\\GripAI_Algo\\Pictures\\Usama\\usama2.png"
    if not os.path.exists(image_path):
        print("Error: Image file does not exist!")
    else:
        print("Processing your hand scan (Diameter-based)...")
        scanner = DiameterHandScanner(image_path)
        scanner.process()