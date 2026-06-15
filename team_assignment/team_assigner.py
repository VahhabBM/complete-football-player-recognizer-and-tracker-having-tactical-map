"""
Team assignment module using K-Means color clustering.

This module assigns football players to teams based on their jersey color.
It uses K-Means clustering to identify two dominant team colors from player
jersey crops, then classifies each player to the nearest team color.

Key features:
- Color extraction from player bounding boxes (jersey region only)
- Green field and dark referee filtering for accurate color detection
- K-Means clustering for robust color space partitioning
- Player history voting system to prevent ID-switching flickering
- Handles edge cases (dark players, poor lighting, shadows)

Algorithm:
1. Extract jersey color from player crops (top 45% to avoid shorts)
2. Filter out field green and referee black colors
3. Apply K-Means on collected colors to identify team clusters
4. Classify new players by nearest cluster center
5. Use 10-frame voting window to smooth predictions
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns football players to teams based on jersey color using K-Means.
    
    Attributes:
        kmeans: K-Means model with 2 clusters (2 teams)
        fitted (bool): Whether model has been trained on color data
        collected_colors (list): Training color samples
        player_history (dict): Per-player prediction history for smoothing
    """

    def __init__(self):
        """Initialize K-Means model and state tracking."""
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=20, max_iter=500)
        self.fitted = False
        self.collected_colors = []
        
        # Track prediction history per player to prevent ID-switching noise
        self.player_history = {} 

    def _get_color(self, frame, bbox):
        """
        Extract representative jersey color from player bounding box.
        
        Args:
            frame: Video frame (BGR image)
            bbox: Bounding box (x1, y1, x2, y2) coordinates
            
        Returns:
            Median color of jersey region, or None if invalid
        """
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        # Handle invalid crops
        if crop.size == 0:
            return None

        # Focus on upper 45% of crop (pure jersey) to avoid shorts noise
        h = crop.shape[0]
        crop = crop[0 : int(h * 0.45), :]

        # Convert to HSV for robust color filtering
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Create mask for grass green (field background)
        lower_green = np.array([35, 35, 35])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_crop, lower_green, upper_green)
        
        # Create mask for dark colors (referee, shadows)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 55])
        dark_mask = cv2.inRange(hsv_crop, lower_dark, upper_dark)

        # Combine masks - exclude grass and dark pixels
        exclude_mask = cv2.bitwise_or(green_mask, dark_mask)
        keep_mask = cv2.bitwise_not(exclude_mask)

        # Extract valid (non-filtered) pixels
        valid_pixels = crop[keep_mask > 0]

        # Fallback: use all pixels if no valid pixels found
        if len(valid_pixels) == 0:
            valid_pixels = crop.reshape(-1, 3)

        # Return median color (robust to outliers)
        return np.median(valid_pixels, axis=0)

    def accumulate_frame_colors(self, frame, bboxes):
        """
        Collect jersey colors from multiple players in a frame.
        
        Args:
            frame: Video frame
            bboxes: List of player bounding boxes
        """
        for bbox in bboxes:
            color = self._get_color(frame, bbox)
            # Only include colors with sufficient brightness to avoid dark artifacts
            if color is not None and np.mean(color) > 45: 
                self.collected_colors.append(color)

    def fit(self):
        """Train K-Means model on collected color samples."""
        if len(self.collected_colors) < 5:
            return
        colors_array = np.array(self.collected_colors)
        self.kmeans.fit(colors_array)
        self.fitted = True
        self.collected_colors = []
        print("[INFO] KMeans model trained with high robustness.")

    def get_team(self, frame, bbox, track_id):
        """
        Predict team for a player using color and history voting.
        
        Args:
            frame: Current video frame
            bbox: Player bounding box
            track_id: Unique player track ID
            
        Returns:
            Team ID (0 or 1), or 0 if unable to determine
        """
        if not self.fitted:
            return 0

        color = self._get_color(frame, bbox)
        if color is None:
            # Fall back to player's previous team if available
            return self.player_history.get(track_id, [0])[-1]

        # Detect and filter referee (dark/black jersey)
        if np.mean(color) < 45:
            return 0

        # Predict team based on current color
        current_prediction = self.kmeans.predict([color])[0]

        # Maintain prediction history per player for voting
        if track_id not in self.player_history:
            self.player_history[track_id] = []
        
        # Store last 10 predictions for majority voting
        self.player_history[track_id].append(current_prediction)
        if len(self.player_history[track_id]) > 10:
            self.player_history[track_id].pop(0)

        # Majority voting: return most common team in last 10 frames
        # This prevents flickering from ID-switching or brief color misdetections
        final_team_id = max(set(self.player_history[track_id]), 
                           key=self.player_history[track_id].count)

        return final_team_id