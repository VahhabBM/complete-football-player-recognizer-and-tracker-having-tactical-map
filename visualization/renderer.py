"""
Rendering and visualization module for football video analysis.

This module handles all visual output including:
- Drawing tracking bounding boxes and keypoints on main video
- Rendering tactical pitch minimap with transformed coordinates
- Creating overlaid minimap in corner of main video

Color scheme:
- Team 0: Orange (team 1)
- Team 1: Blue (team 2)
- Ball: Yellow/Cyan (bright for visibility on green field)
- Goalkeeper: Green
- Referee: Yellow/Cyan
- Pitch: Green field with white lines
"""

import cv2
import numpy as np


class Renderer:
    """
    Video frame rendering for football video analysis.
    
    Attributes:
        team_colors: RGB colors for team 0 and 1
        referee_color: Color for referee players
        goalkeeper_color: Color for goalkeepers
        ball_color: Color for football
        radar_w, radar_h: Minimap dimensions (1050x680)
    """
    
    def __init__(self):
        """Initialize color scheme and minimap dimensions."""
        self.team_colors = {
            0: (255, 120, 0),      # Team 1: Orange
            1: (0, 0, 255)         # Team 2: Blue
        }
        self.referee_color = (0, 255, 255)      # Yellow
        self.goalkeeper_color = (0, 255, 0)     # Green
        self.ball_color = (0, 255, 255)         # Bright yellow-cyan for visibility

        self.radar_w = 1050
        self.radar_h = 680

    def draw_tracks(self, frame, detections):
        """
        Draw detection bounding boxes and labels on main video frame.
        
        Args:
            frame: Video frame to draw on
            detections: List of detection dictionaries with bbox, class_id, team_id
            
        Returns:
            Frame with drawn detections
        """
        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            x1, y1, x2, y2 = map(int, bbox)

            if class_id == 0:  # Ball: white circle
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 6, (255, 255, 255), -1)
            elif class_id == 1:  # Goalkeeper: green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.goalkeeper_color, 2)
            elif class_id == 2:  # Player: color-coded by team
                team_id = det.get("team_id", 0)
                color = self.team_colors[team_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            elif class_id == 3:  # Referee: yellow rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.referee_color, 2)
        return frame

    def draw_keypoints(self, frame, keypoints):
        """
        Draw pose keypoints on main video frame.
        
        Only draws selected football pitch keypoints for clarity:
        - Indices: 13, 14, 15, 16 (center vertical)
        - Indices: 17, 23, 24, 30, 31 (side points)
        
        Args:
            frame: Video frame to draw on
            keypoints: YOLO pose keypoint detection result
            
        Returns:
            Frame with drawn keypoints
        """
        if keypoints is None or not hasattr(keypoints, 'xy') or len(keypoints.xy) == 0:
            return frame
        allowed_indices = [13, 14, 15, 16, 17, 23, 24, 30, 31]
        
        kp_xy = keypoints.xy[0].cpu().numpy()
        for idx, kp in enumerate(kp_xy):
            if idx in allowed_indices:
                x, y = map(int, kp)
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        return frame

    def draw_tactical_pitch(self, detections):
        """
        Create a tactical view (minimap) of the field with player positions.
        
        Draws:
        - Green field background
        - White pitch lines and markings
        - Center circle and spot
        - Goal boxes
        - Player positions as colored circles
        - Ball position as yellow circle
        
        Args:
            detections: List of detections with transformed_coords
            
        Returns:
            Minimap image array (1050x680)
        """
        # Create green field canvas
        pitch = np.zeros((self.radar_h, self.radar_w, 3), dtype=np.uint8)
        pitch[:] = (35, 115, 35)  # Green field

        w, h = self.radar_w, self.radar_h
        mx, my = w // 2, h // 2

        # Draw pitch outline
        cv2.rectangle(pitch, (20, 20), (w - 20, h - 20), (255, 255, 255), 3)
        
        # Draw center line
        cv2.line(pitch, (mx, 20), (mx, h - 20), (255, 255, 255), 3)
        
        # Draw center circle and center spot
        cv2.circle(pitch, (mx, my), 92, (255, 255, 255), 3)
        cv2.circle(pitch, (mx, my), 5, (255, 255, 255), -1)

        # Draw goal boxes (left side)
        cv2.rectangle(pitch, (20, my - 150), (185, my + 150), (255, 255, 255), 3)
        cv2.rectangle(pitch, (20, my - 60), (75, my + 60), (255, 255, 255), 2)
        
        # Draw goal boxes (right side)
        cv2.rectangle(pitch, (w - 185, my - 150), (w - 20, my + 150), (255, 255, 255), 3)
        cv2.rectangle(pitch, (w - 75, my - 60), (w - 20, my + 60), (255, 255, 255), 2)

        # Draw player positions
        for det in detections:
            if "transformed_coords" in det and det["transformed_coords"] is not None:
                tx, ty = det["transformed_coords"]
                class_id = det["class_id"]

                # Only draw if within minimap bounds
                if -10 <= tx <= w + 10 and -10 <= ty <= h + 10:
                    if class_id == 0:    # Ball: bright yellow with black border
                        cv2.circle(pitch, (tx, ty), 12, self.ball_color, -1)
                        cv2.circle(pitch, (tx, ty), 12, (0, 0, 0), 3)
                    elif class_id == 1:  # Goalkeeper: green
                        cv2.circle(pitch, (tx, ty), 12, self.goalkeeper_color, -1)
                        cv2.circle(pitch, (tx, ty), 12, (255, 255, 255), 2)
                    elif class_id == 2:  # Player: team colored
                        team_id = det.get("team_id", 0)
                        color = self.team_colors[team_id]
                        cv2.circle(pitch, (tx, ty), 12, color, -1)
                        cv2.circle(pitch, (tx, ty), 12, (255, 255, 255), 2)
        return pitch

    def add_minimap_to_frame(self, main_frame, tactical_pitch, opacity=0.75):
        """
        Overlay tactical minimap in bottom-right corner of main frame.
        
        Args:
            main_frame: Main video frame
            tactical_pitch: Minimap image (1050x680)
            opacity: Alpha transparency for overlay (0.75 = 75% minimap, 25% frame)
            
        Returns:
            Main frame with overlaid minimap
        """
        # Resize minimap to bottom-right corner size (360x233)
        target_w = 360
        target_h = 233
        resized_pitch = cv2.resize(tactical_pitch, (target_w, target_h), 
                                   interpolation=cv2.INTER_AREA)

        # Position in bottom-right with 30px margin
        offset_x = main_frame.shape[1] - target_w - 30
        offset_y = main_frame.shape[0] - target_h - 30

        # Blend minimap with frame using alpha transparency
        roi = main_frame[offset_y:offset_y+target_h, offset_x:offset_x+target_w]
        blended_roi = cv2.addWeighted(resized_pitch, opacity, roi, 1.0 - opacity, 0)
        main_frame[offset_y:offset_y+target_h, offset_x:offset_x+target_w] = blended_roi
        
        # Draw white border around minimap
        cv2.rectangle(main_frame, (offset_x, offset_y), 
                     (offset_x+target_w, offset_y+target_h), (255, 255, 255), 1)
        return main_frame