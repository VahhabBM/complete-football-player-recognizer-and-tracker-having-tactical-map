"""
Object tracking module for multi-object tracking in video frames.

This module implements video object tracking using YOLO object detection 
combined with ByteTrack algorithm for frame-by-frame tracking. It maintains
consistent tracking IDs across frames for the same objects.

Key components:
- YOLO model: Performs object detection in each frame
- ByteTrack: Associates detections across frames to maintain track IDs
- Tracker class: Integrates detection and tracking operations

Output format:
- List of frame-level tracks with detection information
- Each track contains: bbox, class_id, confidence, track_id
"""

from ultralytics import YOLO
import supervision as sv


class Tracker:
    """
    Video object tracker combining YOLO detection with ByteTrack association.
    
    Attributes:
        model: Loaded YOLO detection model
        tracker: ByteTrack instance for multi-object tracking
    """

    def __init__(self, model_path):
        """
        Initialize tracker with detection model.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_and_track(self, frames):
        """
        Detect objects and track them across video frames.
        
        Args:
            frames (list): List of video frames (numpy arrays)
            
        Returns:
            list: Tracks for each frame containing detection dictionaries
                  with keys: bbox, class_id, confidence, track_id
        """
        tracks = []

        for frame in frames:
            # YOLO object detection with 30% confidence threshold
            result = self.model.predict(
                frame,
                conf=0.3,
                verbose=False
            )[0]

            # Convert YOLO detections to supervision format
            detections = sv.Detections.from_ultralytics(result)

            # Update tracker with new detections and get tracked objects
            tracked_detections = self.tracker.update_with_detections(
                detections
            )

            frame_tracks = []

            # Convert tracked detections to clean dictionary format
            for i in range(len(tracked_detections)):
                bbox = tracked_detections.xyxy[i]
                confidence = tracked_detections.confidence[i]
                class_id = int(tracked_detections.class_id[i])
                track_id = tracked_detections.tracker_id[i]

                # Skip objects without valid track ID
                if track_id is None:
                    continue

                frame_tracks.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "confidence": confidence,
                    "track_id": int(track_id)
                })

            tracks.append(frame_tracks)

        return tracks