import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera import Camera
from vision.face_tracker import FaceTracker
from vision.hand_tracker import HandTracker
from interaction.cigarette_tracker import CigaretteTracker
from effects.cigarette import CigaretteRenderer, CigaretteRendererFallback


def main():
    camera = Camera(device_index=0, width=1280, height=720)
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    cigarette_tracker = CigaretteTracker()
    cigarette_renderer = CigaretteRenderer()

    fallback_renderer = CigaretteRendererFallback()

    try:
        camera.open()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    print("Virtual Smoking - Phase 5: Virtual Cigarette Tracking")
    print("Controls: 'q' or ESC to quit, 'd' to toggle debug landmarks, 'c' to toggle cigarette debug")
    print()

    show_debug = True
    show_cigarette_debug = False

    while True:
        frame = camera.read()
        if frame is None:
            print("Failed to read frame")
            break

        face_detected = face_tracker.process(frame)
        hand_detected = hand_tracker.process(frame)

        hand_landmarks = hand_tracker.get_all_landmarks()
        cigarette_tracker.update(hand_landmarks)

        if show_debug:
            face_tracker.draw_landmarks(frame)
            hand_tracker.draw_landmarks(frame)

        # Render cigarette
        if cigarette_tracker.is_held and cigarette_tracker.position is not None:
            if cigarette_renderer.cigarette_img is not None:
                cigarette_renderer.draw(frame, cigarette_tracker.position, cigarette_tracker.rotation, 0.0)
            else:
                fallback_renderer.draw(frame, cigarette_tracker.position, cigarette_tracker.rotation, 0.0)

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
        cv2.putText(frame, f"Cigarette: {'HELD' if cigarette_tracker.is_held else 'NOT HELD'}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 0) if cigarette_tracker.is_held else (0, 0, 255), 2)
        status_y += 30
        cv2.putText(frame, f"FPS: {camera.get_fps():.1f}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if show_cigarette_debug and cigarette_tracker.is_held:
            debug = cigarette_tracker.get_debug_info()
            status_y += 30
            cv2.putText(frame, f"Cig Pos: ({debug['position'][0]:.0f}, {debug['position'][1]:.0f})" if debug['position'] else "Cig Pos: N/A",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            status_y += 25
            cv2.putText(frame, f"Cig Angle: {debug['rotation_deg']:.1f} deg",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            status_y += 25
            cv2.putText(frame, f"Raw Pos: ({debug['raw_position'][0]:.0f}, {debug['raw_position'][1]:.0f})" if debug['raw_position'] else "Raw Pos: N/A",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        status_y += 30
        cv2.putText(frame, "Press 'q' to quit, 'd' to toggle debug, 'c' for cig debug",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Virtual Smoking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('c'):
            show_cigarette_debug = not show_cigarette_debug

    camera.close()
    face_tracker.close()
    hand_tracker.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())