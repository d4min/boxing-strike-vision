"""
BoxingVisionStrike: Computer vision system for analysing boxing punches.
"""

import cv2
import time 

class VideoStream:
    """
    Handles webcam capture and basic frame processing.
    """

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Initialise the video capture stream.

        Args:
            camera_id: ID of the camera to use (default: 0)
            width: Frame width (default: 640)
            height: Frame height (default: 480)
            fps: Target frames per second (default: 30)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    def start(self):
        """
        Start the video stream.
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {self.camera_id}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        return self

    def read(self):
        """
        Read a frame from the video stream.
        """
        if self.cap is None:
            raise RuntimeError("Video stream not started. Call start() first.")
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None
            
        # Mirror the frame for more intuitive display
        return cv2.flip(frame, 1)
    
    def release(self):
        """
        Release the video capture resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class FPSCounter:
    """
    Tracks and displays frames per second.
    """
    def __init__(self):
        self.prev_time = 0
        self.current_time = 0
        self.fps = 0
    
    def update(self):
        """Update FPS calculation."""
        self.current_time = time.time()
        time_diff = self.current_time - self.prev_time
        
        # Avoid division by zero
        if time_diff > 0:
            self.fps = 1.0 / time_diff
            
        self.prev_time = self.current_time
        return self.fps
    
    def draw(self, frame):
        """Draw FPS information on the frame."""
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(
            frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        return frame
    
class BoxingStrikeVision:
    """
    Main application class that coordinates the different components.
    """
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """
        Initialize the PunchTracker application.
        
        Args:
            camera_id: ID of the camera to use (default: 0)
            width: Frame width (default: 640)
            height: Frame height (default: 480)
            fps: Target frames per second (default: 30)
        """
        self.video_stream = VideoStream(camera_id, width, height, fps)
        self.fps_counter = FPSCounter()
        self.running = False
        self.window_name = "PunchTracker"
    
    def setup(self):
        """Prepare the application for running."""
        self.video_stream.start()
        cv2.namedWindow(self.window_name)
        print("PunchTracker initialized. Press 'q' to quit.")
        return self
    
    def process_frame(self, frame):
        """
        Process a video frame.
        This method will be expanded in future steps to include pose detection and punch analysis.
        """
        if frame is None:
            return None
        
        # Update and display FPS
        self.fps_counter.update()
        frame = self.fps_counter.draw(frame)
        
        return frame
    
    def run(self):
        """Main application loop."""
        self.running = True
        
        while self.running:
            # Capture frame
            frame = self.video_stream.read()
            
            if frame is None:
                print("Failed to grab frame. Exiting...")
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the result
            cv2.imshow(self.window_name, processed_frame)
            
            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
    
    def cleanup(self):
        """Release resources and perform cleanup."""
        self.video_stream.release()
        cv2.destroyAllWindows()
        print("BoxingStrikeVision shutdown complete.")

def main():
    """Entry point for the application."""
    tracker = BoxingStrikeVision()
    
    try:
        tracker.setup().run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tracker.cleanup()


if __name__ == "__main__":
    main()