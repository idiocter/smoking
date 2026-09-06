import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera import Camera
from vision.face_tracker import FaceTracker
from vision.hand_tracker import HandTracker
from interaction.cigarette_tracker import CigaretteTracker
from interaction.cigarette_mouth_detector import CigaretteMouthDetector, CigaretteMouthState
from interaction.smoking_detector import SmokingDetector, SmokingState
from effects.cigarette import CigaretteRenderer, CigaretteRendererFallback
from effects.glow import GlowEffect
from effects.smoke import SmokeEffect


def main():
    camera = Camera(device_index=0, width=1280, height=720)
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    cigarette_tracker = CigaretteTracker()
    cigarette_mouth_detector = CigaretteMouthDetector()
    smoking_detector = SmokingDetector()
    cigarette_renderer = CigaretteRenderer()
    glow_effect = GlowEffect()
    smoke_effect = SmokeEffect()

    fallback_renderer = CigaretteRendererFallback()

    try:
        camera.open()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    print("Virtual Smoking - Phase 10: Real-Time Virtual Smoke Particle Effect")
    print("Controls: 'q' or ESC to quit, 'd' landmarks, 'c' cigarette, 'i' interaction, 's' smoking, 'g' glow, 'k' smoke")
    print()

    show_debug = True
    show_cigarette_debug = False
    show_interaction_debug = True
    show_smoking_debug = True
    show_glow_debug = True
    show_smoke_debug = True

    while True:
        frame = camera.read()
        if frame is None:
            print("Failed to read frame")
            break

        face_detected = face_tracker.process(frame)
        hand_detected = hand_tracker.process(frame)

        hand_landmarks = hand_tracker.get_all_landmarks()
        cigarette_tracker.update(hand_landmarks)

        interaction_state = CigaretteMouthState.FAR
        interaction_distance = None
        mouth_center = None
        smoking_state = SmokingState.IDLE
        pattern_detected = False
        exhalation_detected = False

        if face_detected and cigarette_tracker.is_held:
            mouth_center = face_tracker.get_mouth_center()
            interaction_state = cigarette_mouth_detector.update(cigarette_tracker, face_tracker)
            interaction_distance = cigarette_mouth_detector.get_distance()
            smoking_state = smoking_detector.update(cigarette_tracker, cigarette_mouth_detector, face_tracker)
            pattern_detected = smoking_detector.is_pattern_detected()
            exhalation_detected = smoking_detector.is_exhalation_detected()

        is_inhaling = (smoking_state == SmokingState.INHALING)
        glow_effect.set_target(is_inhaling)
        glow_effect.update()

        # Update smoke effect with exhalation detection and mouth position
        smoke_effect.update(exhalation_detected, mouth_center)

        if show_debug:
            face_tracker.draw_landmarks(frame)
            hand_tracker.draw_landmarks(frame)

        if cigarette_tracker.is_held and cigarette_tracker.position is not None:
            if cigarette_renderer.cigarette_img is not None:
                cigarette_renderer.draw(frame, cigarette_tracker.position, cigarette_tracker.rotation, 0.0)
            else:
                fallback_renderer.draw(frame, cigarette_tracker.position, cigarette_tracker.rotation, 0.0)

            glow_effect.draw(frame, cigarette_tracker.position, cigarette_tracker.rotation, cigarette_tracker.length)

        # Render smoke particles
        smoke_effect.draw(frame)

        if show_interaction_debug and mouth_center and cigarette_tracker.is_held:
            cig_pos = cigarette_tracker.get_mouth_end_position(mouth_center)
            if cig_pos:
                cv2.line(frame,
                        (int(cig_pos[0]), int(cig_pos[1])),
                        (int(mouth_center[0]), int(mouth_center[1])),
                        (0, 255, 255), 2)
                cv2.circle(frame, (int(mouth_center[0]), int(mouth_center[1])), 6, (0, 255, 255), 2)
                cv2.circle(frame, (int(cig_pos[0]), int(cig_pos[1])), 6, (255, 0, 255), -1)

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

        state_color = (255, 255, 255)
        if interaction_state == CigaretteMouthState.NEAR:
            state_color = (0, 255, 0)
        elif interaction_state == CigaretteMouthState.APPROACHING:
            state_color = (0, 255, 255)
        elif interaction_state == CigaretteMouthState.MOVING_AWAY:
            state_color = (0, 165, 255)
        elif interaction_state == CigaretteMouthState.FAR:
            state_color = (100, 100, 100)

        cv2.putText(frame, f"Cig State: {interaction_state}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        status_y += 30

        if interaction_distance is not None:
            cv2.putText(frame, f"Mouth Distance: {interaction_distance:.1f} px",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status_y += 30

        cv2.putText(frame, f"FPS: {camera.get_fps():.1f}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if show_interaction_debug:
            debug = cigarette_mouth_detector.get_debug_info()
            status_y += 30
            cv2.putText(frame, f"Threshold: {debug['threshold']} px",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25
            cv2.putText(frame, f"Frames in state: {debug['frames_in_state']}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if show_smoking_debug:
            smoking_debug = smoking_detector.get_debug_info()
            status_y += 30

            smoke_color = (255, 255, 255)
            if smoking_state == SmokingState.INHALING:
                smoke_color = (0, 255, 0)
            elif smoking_state == SmokingState.EXHALING:
                smoke_color = (255, 0, 255)
            elif smoking_state == SmokingState.EXHALATION_CANDIDATE:
                smoke_color = (255, 100, 255)
            elif smoking_state == SmokingState.INHALATION_CANDIDATE:
                smoke_color = (0, 255, 255)
            elif smoking_state == SmokingState.NEAR_MOUTH:
                smoke_color = (0, 200, 255)
            elif smoking_state == SmokingState.APPROACHING:
                smoke_color = (0, 165, 255)
            elif smoking_state == SmokingState.COMPLETED:
                smoke_color = (0, 255, 0)
            elif smoking_state == SmokingState.IDLE:
                smoke_color = (100, 100, 100)

            cv2.putText(frame, f"Smoking State: {smoking_state}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, smoke_color, 2)
            status_y += 30

            if pattern_detected:
                cv2.putText(frame, "Inhalation Pattern: DETECTED",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                status_y += 30
            else:
                cv2.putText(frame, "Inhalation Pattern: NOT DETECTED",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                status_y += 25

            if exhalation_detected:
                cv2.putText(frame, "Exhalation Pattern: DETECTED",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                status_y += 30
            else:
                cv2.putText(frame, "Exhalation Pattern: NOT DETECTED",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                status_y += 25

            cv2.putText(frame, f"Frames in state: {smoking_debug['frames_in_state']}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25

            cv2.putText(frame, f"Inhalation Done: {'YES' if smoking_debug.get('inhalation_completed', False) else 'NO'}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25

            th = smoking_debug['thresholds']
            cv2.putText(frame, f"Thresh: near={th['near_mouth']} open_chg={th['mouth_opening_change']} ar_chg={th['mouth_ar_change']:.1f} exh_open={th.get('exhalation_opening', 0)} exh_w={th.get('exhalation_width_change', 0)}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            status_y += 18

            fc = smoking_debug['frame_counts']
            cv2.putText(frame, f"Frames: appr={fc['approaching']} near={fc['near_mouth']} inh={fc['inhalation']} away={fc['away']} exh={fc.get('exhalation', 0)}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        if show_glow_debug:
            status_y += 30
            cv2.putText(frame, f"Glow Intensity: {glow_effect.get_intensity():.2f}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_y += 25
            cv2.putText(frame, f"Glow Target: {'ON' if is_inhaling else 'OFF'}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25
            cv2.putText(frame, f"Fade In: {glow_effect.fade_in_speed} Fade Out: {glow_effect.fade_out_speed}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        if show_smoke_debug:
            status_y += 30
            cv2.putText(frame, f"Smoke: {'ACTIVE' if smoke_effect.is_active() else 'INACTIVE'}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            status_y += 25
            cv2.putText(frame, f"Particles: {smoke_effect.get_particle_count()}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25
            cv2.putText(frame, f"Exhalation Triggered: {'YES' if smoke_effect.exhalation_triggered else 'NO'}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if show_cigarette_debug and cigarette_tracker.is_held:
            cig_debug = cigarette_tracker.get_debug_info()
            status_y += 30
            cv2.putText(frame, f"Cig Pos: ({cig_debug['position'][0]:.0f}, {cig_debug['position'][1]:.0f})" if cig_debug['position'] else "Cig Pos: N/A",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            status_y += 25
            cv2.putText(frame, f"Cig Angle: {cig_debug['rotation_deg']:.1f} deg",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if face_detected and (show_interaction_debug or show_smoking_debug):
            mouth_data = face_tracker.get_mouth_measurements()
            status_y += 30
            cv2.putText(frame, f"Mouth W: {mouth_data['width']:.1f} H: {mouth_data['height']:.1f} Open: {mouth_data['opening']:.1f} AR: {mouth_data['aspect_ratio']:.2f}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        status_y += 30
        cv2.putText(frame, "Press 'q' quit, 'd' landmarks, 'c' cig, 'i' inter, 's' smoke, 'g' glow, 'k' smoke",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Virtual Smoking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('c'):
            show_cigarette_debug = not show_cigarette_debug
        elif key == ord('i'):
            show_interaction_debug = not show_interaction_debug
        elif key == ord('s'):
            show_smoking_debug = not show_smoking_debug
        elif key == ord('g'):
            show_glow_debug = not show_glow_debug
        elif key == ord('k'):
            show_smoke_debug = not show_smoke_debug

    camera.close()
    face_tracker.close()
    hand_tracker.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())