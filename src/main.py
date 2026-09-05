import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera import Camera
from vision.face_tracker import FaceTracker
from vision.hand_tracker import HandTracker


def main():
    camera = Camera(device_index=0, width=1280, height=720)
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()

    try:
        camera.open()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    print("Virtual Smoking - Phase 4: Combined Tracking")
    print("Controls: 'q' or ESC to quit, 'd' to toggle debug landmarks")
    print()

    show_debug = True

    while True:
        frame = camera.read()
        if frame is None:
            print("Failed to read frame")
            break

        face_detected = face_tracker.process(frame)
        hand_detected = hand_tracker.process(frame)

        if show_debug:
            face_tracker.draw_landmarks(frame)
            hand_tracker.draw_landmarks(frame)

        h, w = frame.shape[:2]
        status_y = 30
        cv2.putText(frame, f"Face: {'DETECTED' if face_detected else 'NOT DETECTED'}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 0) if face_detected else (0, 0, 255), 2)
        status_y += 30
        cv2.putText(frame, f"Hand: {'DETECTED' if hand_detected else 'NOT DETECTED'}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 0) if hand_detected else (0, 0, 255), 2)
        status_y += 30
        cv2.putText(frame, f"FPS: {camera.get_fps():.1f}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(frame, "Press 'q' to quit, 'd' to toggle debug",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Virtual Smoking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('d'):
            show_debug = not show_debug

    camera.close()
    face_tracker.close()
    hand_tracker.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())