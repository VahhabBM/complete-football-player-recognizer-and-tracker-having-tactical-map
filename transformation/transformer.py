"""
Field view transformation module using pose keypoints.

This module computes homography matrices to transform player and ball 
positions from the camera view to a standardized tactical field view 
(minimap). It uses YOLO pose detection to identify football pitch 
keypoints, then computes perspective transformation matrices.

Key features:
- Detects 17 COCO keypoints on the football pitch
- Computes homography using center and side pitch keypoints
- Validates and corrects homography with 180° rotation + vertical flip
- Implements 2-stage estimation: center lines first, then side lines
- Buffers last valid homography for robustness against missing keypoints
- Transforms player/ball positions to 1050x680 minimap coordinates

Tactical field dimensions: 1050x680 pixels (represents 1 pitch)
"""

import cv2
import numpy as np
from ultralytics import YOLO


class Transformer:
    """
    Transforms camera view coordinates to standardized tactical field view.
    
    Attributes:
        model: YOLO pose model for field keypoint detection
        conf_threshold: Confidence threshold for keypoint validation
        last_valid_H: Last validated homography matrix (fallback)
        radar_w, radar_h: Minimap dimensions (1050x680)
        center_pitch_kps: Center line keypoint mappings
        side_pitch_kps: Side line keypoint mappings
    """
    
    def __init__(self, model_path, conf_threshold=0.45): 
        """
        Initialize pose-based field transformer.
        
        Args:
            model_path (str): Path to YOLOv8 pose model weights
            conf_threshold (float): Minimum keypoint confidence (0.45)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.last_valid_H = None  # Fallback for tracking continuity

        # Minimap dimensions representing the entire field
        self.radar_w = 1050
        self.radar_h = 680

        # Center line keypoints (vertical backbone of field)
        # Maps COCO keypoint indices to minimap pixel coordinates
        self.center_pitch_kps = {
            13: [525, 0],      # Top center
            14: [525, 248],    # Upper-middle center
            15: [525, 432],    # Lower-middle center
            16: [525, 680],    # Bottom center
            31: [433, 340],    # Left middle
            30: [617, 340]     # Right middle
        }

        # Right side keypoints for validation (extends center estimate)
        self.side_pitch_kps = {
            24: [1050, 0],     # Top right corner
            17: [885, 138],    # Right side upper
            23: [1050, 680]    # Bottom right corner
        }

    
    def detect_keypoints(self, frame):
        """
        Detect football pitch keypoints using YOLO pose model.
        
        Args:
            frame: Video frame
            
        Returns:
            Keypoints object with detected pose, or None if none detected
        """
        result = self.model.predict(frame, conf=0.15, imgsz=640, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        return result.keypoints

    def fix_and_validate_homography(self, H):
        """
        Validate and correct homography matrix with mandatory transformations.
        
        Applies:
        1. 180° rotation (fixes field orientation)
        2. Vertical flip (aligns field width)
        3. Determinant validation (checks matrix structure integrity)
        4. Bounding box validation (ensures full field in bounds)
        
        Args:
            H: Homography matrix to validate
            
        Returns:
            Valid corrected homography, or None if invalid
        """
        if H is None:
            return None
        try:
            # Validate determinant to ensure matrix is not collapsed
            det = np.linalg.det(H[:2, :2])
            if abs(det) < 1e-3 or abs(det) > 1e3:
                return None  

            # Check if transformed image bounds are reasonable
            w, h = 1920, 1080
            src_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(src_corners, H).reshape(-1, 2)

            # Reject homographies that collapse the field to center line
            min_x, max_x = np.min(dst_corners[:, 0]), np.max(dst_corners[:, 0])
            min_y, max_y = np.min(dst_corners[:, 1]), np.max(dst_corners[:, 1])
            if (max_x - min_x) < 300 or (max_y - min_y) < 200: 
                return None  

            # 1. Mandatory 180° rotation (corrects field orientation)
            M = cv2.getRotationMatrix2D((self.radar_w / 2, self.radar_h / 2), 180, 1.0)
            R = np.eye(3)
            R[:2, :] = M
            H = np.dot(R, H)

            # 2. Mandatory vertical flip (aligns field width correctly)
            R_flip = np.array([
                [1,  0, 0],
                [0, -1, self.radar_h],
                [0,  0, 1]
            ], dtype=np.float32)
            H = np.dot(R_flip, H)

            return H
        except:
            return None

    def get_homography_matrix(self, keypoints_object):
        """
        Compute homography matrix from detected keypoints.
        
        Two-stage strategy:
        1. Use center keypoints (most stable)
        2. Validate and add side keypoints if within 60px error
        3. Fallback to all keypoints if center-only fails
        
        Args:
            keypoints_object: YOLO keypoint detection result
            
        Returns:
            Homography matrix (3x3), or last valid matrix if current is bad
        """
        if keypoints_object is None or not hasattr(keypoints_object, 'xy') or len(keypoints_object.xy) == 0:
            return self.last_valid_H

        # Extract keypoint positions and confidences
        kp_xy = keypoints_object.xy[0].cpu().numpy()
        if hasattr(keypoints_object, 'conf') and keypoints_object.conf is not None:
            kp_conf = keypoints_object.conf[0].cpu().numpy()
        else:
            kp_conf = np.ones(len(kp_xy))

        center_src, center_dst = [], []
        side_src, side_dst = [], []

        # Collect high-confidence keypoints for center and side lines
        for idx, (kp, conf) in enumerate(zip(kp_xy, kp_conf)):
            x, y = float(kp[0]), float(kp[1])
            if x > 0 and y > 0 and conf >= self.conf_threshold:
                if idx in self.center_pitch_kps:
                    center_src.append([x, y])
                    center_dst.append(self.center_pitch_kps[idx])
                elif idx in self.side_pitch_kps:
                    side_src.append([x, y])
                    side_dst.append(self.side_pitch_kps[idx])

        new_H = None

        # Need minimum 4 points for homography computation
        if (len(center_src) + len(side_src)) >= 4:
            # Stage 1: Primary homography from center points
            if len(center_src) >= 4:
                c_src = np.array(center_src, dtype=np.float32)
                c_dst = np.array(center_dst, dtype=np.float32)
                base_H, _ = cv2.findHomography(c_src, c_dst, cv2.RANSAC, 5.0)
                base_H = self.fix_and_validate_homography(base_H)
                
                final_src = list(center_src)
                final_dst = list(center_dst)
                
                # Stage 2: Validate and add side points if they align well
                if base_H is not None and len(side_src) > 0:
                    s_pts = np.array(side_src, dtype=np.float32).reshape(-1, 1, 2)
                    transformed_sides = cv2.perspectiveTransform(s_pts, base_H).reshape(-1, 2)
                    
                    for i, (tx, ty) in enumerate(transformed_sides):
                        actual_target = self.side_pitch_kps[list(self.side_pitch_kps.keys())[i]]
                        distance = np.sqrt((tx - actual_target[0])**2 + (ty - actual_target[1])**2)
                        if distance < 60:  # Accept if within 60px error
                            final_src.append(side_src[i])
                            final_dst.append(actual_target)

                # Final homography with all validated points
                f_src = np.array(final_src, dtype=np.float32)
                f_dst = np.array(final_dst, dtype=np.float32)
                new_H, _ = cv2.findHomography(f_src, f_dst, cv2.RANSAC, 5.0)
            
            # Fallback: compute from all available points
            if new_H is None:
                all_src = center_src + side_src
                all_dst = center_dst + side_dst
                f_src = np.array(all_src, dtype=np.float32)
                f_dst = np.array(all_dst, dtype=np.float32)
                new_H, _ = cv2.findHomography(f_src, f_dst, cv2.RANSAC, 5.0)

        # Final validation and buffer update
        validated_H = self.fix_and_validate_homography(new_H)
        if validated_H is not None:
            self.last_valid_H = validated_H
        
        return self.last_valid_H

    def transform_position(self, H, bbox, is_ball=False):
        """
        Transform bounding box center/foot point to minimap coordinates.
        
        Args:
            H: Homography matrix
            bbox: Bounding box (x1, y1, x2, y2)
            is_ball: If True, use center; if False, use bottom center (foot)
            
        Returns:
            Tuple (tx, ty) minimap coordinates, or None if out of bounds
        """
        if H is None:
            return None

        x1, y1, x2, y2 = bbox
        
        # Ball: center of bounding box
        # Player: bottom center (foot position for tactical relevance)
        if is_ball:
            foot_x = (x1 + x2) / 2
            foot_y = (y1 + y2) / 2
        else:
            foot_x = (x1 + x2) / 2
            foot_y = y2

        # Apply perspective transformation
        point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, H)

        tx = int(transformed[0][0][0])
        ty = int(transformed[0][0][1])

        # Validate position is within minimap bounds (with ±30px tolerance)
        if -30 <= tx <= self.radar_w + 30 and -30 <= ty <= self.radar_h + 30:
            return (tx, ty)
        return None