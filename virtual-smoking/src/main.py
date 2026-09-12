import argparse
import cv2
import sys
import os
from pathlib import Path

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
from config import Config

try:
    from effects.cigarette_3d import Cigarette3DRenderer
    CIGARETTE_3D_IMPORT_ERROR = None
except ImportError as exc:
    Cigarette3DRenderer = None
    CIGARETTE_3D_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / 'assets' / 'cigarette'
FACE_MODEL_PATH = PROJECT_ROOT / 'face_landmarker.task'
HAND_MODEL_PATH = PROJECT_ROOT / 'hand_landmarker.task'


def validate_assets(project_root=PROJECT_ROOT):
    """Verify required assets exist at startup."""
    project_root = Path(project_root)
    assets = [
        project_root / 'face_landmarker.task',
        project_root / 'hand_landmarker.task',
        project_root / 'assets' / 'cigarette' / 'cigarette.png',
        project_root / 'assets' / 'cigarette' / 'cigarette_glow.png',
    ]
    missing = [asset for asset in assets if not asset.is_file()]
    if missing:
        print("ERROR: Missing required assets:")
        for asset in missing:
            print(f"  {asset}")
        print("Follow the model download instructions in README.md.")
        return False
    return True


class VirtualSmokingApp:
    def __init__(self, debug_mode=False, prefer_3d=True):
        self.debug_mode = debug_mode
        self.camera = None
        self.face_tracker = None
        self.hand_tracker = None
        self.cigarette_tracker = None
        self.cigarette_mouth_detector = None
        self.smoking_detector = None
        self.cigarette_renderer_3d = None
        self.glow_effect = None
        self.smoke_effect = None
        self.fallback_renderer = None
        self.running = False
        self.prefer_3d = prefer_3d
        self.use_3d = False

    def initialize(self):
        """Initialize all components."""
        # Validate assets first
        if not validate_assets():
            return False

        try:
            self.camera = Camera(
                device_index=Config.CAMERA['device_index'],
                width=Config.CAMERA['width'],
                height=Config.CAMERA['height'],
                fps=Config.CAMERA['fps'],
            )
            self.face_tracker = FaceTracker(
                max_faces=Config.FACE_TRACKER['max_faces'],
                min_detection_confidence=Config.FACE_TRACKER['min_detection_confidence'],
                min_tracking_confidence=Config.FACE_TRACKER['min_tracking_confidence'],
                model_asset_path=FACE_MODEL_PATH,
            )
            self.hand_tracker = HandTracker(
                max_hands=Config.HAND_TRACKER['max_hands'],
                min_detection_confidence=Config.HAND_TRACKER['min_detection_confidence'],
                min_tracking_confidence=Config.HAND_TRACKER['min_tracking_confidence'],
                model_asset_path=HAND_MODEL_PATH,
            )
            self.cigarette_tracker = CigaretteTracker()
            self.cigarette_mouth_detector = CigaretteMouthDetector()
            self.smoking_detector = SmokingDetector()

            # Initialize 3D cigarette renderer
            model_path = ASSET_DIR / 'cigarette.glb'
            if self.prefer_3d and Cigarette3DRenderer is None:
                print(f"Warning: 3D dependencies unavailable; using 2D renderer: {CIGARETTE_3D_IMPORT_ERROR}")
            elif self.prefer_3d and not model_path.is_file():
                print(f"Warning: 3D model not found at {model_path}; using 2D renderer")
            elif self.prefer_3d:
                try:
                    renderer_config = Config.CIGARETTE_3D.copy()
                    renderer_config.update({
                        'glow_fade_in': Config.GLOW_EFFECT['fade_in_speed'],
                        'glow_fade_out': Config.GLOW_EFFECT['fade_out_speed'],
                    })
                    self.cigarette_renderer_3d = Cigarette3DRenderer(
                        model_path,
                        config=renderer_config,
                    )
                    self.use_3d = True
                    print("3D cigarette renderer initialized successfully")
                except Exception as exc:
                    print(f"Warning: 3D renderer failed to initialize; using 2D renderer: {exc}")
                    self.cigarette_renderer_3d = None
            
            # Fallback 2D renderer
            self.cigarette_renderer = CigaretteRenderer(ASSET_DIR / 'cigarette.png')
            self.fallback_renderer = CigaretteRendererFallback()
            
            # Glow effect (for 2D fallback)
            self.glow_effect = GlowEffect(
                asset_path=ASSET_DIR / 'cigarette_glow.png',
                max_intensity=Config.GLOW_EFFECT['max_intensity'],
                fade_in_speed=Config.GLOW_EFFECT['fade_in_speed'],
                fade_out_speed=Config.GLOW_EFFECT['fade_out_speed']
            )
            
            self.smoke_effect = SmokeEffect()

            self.camera.open()
            return True

        except RuntimeError as e:
            print(f"Initialization error: {e}")
            self.shutdown()
            return False
        except Exception as e:
            print(f"Unexpected initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.shutdown()
            return False

    def run(self):
        """Main processing loop."""
        if not self.running:
            self.running = True

        print("Virtual Smoking - Ready")
        if self.debug_mode:
            print("Debug mode: ON")
            print("Controls: 'q'/'ESC' quit, 'D' toggle debug, '3' toggle 3D/2D")
        else:
            print("Controls: 'q'/'ESC' to quit, 'D' to enable debug mode")
        print()

        while self.running:
            frame = self.camera.read()
            if frame is None:
                print("Failed to read frame")
                break

            h, w = frame.shape[:2]
            
            # Initialize 3D renderer projection on first frame
            if self.use_3d and self.cigarette_renderer_3d:
                self.cigarette_renderer_3d.set_view_projection(w, h)

            try:
                face_detected = self.face_tracker.process(frame)
            except Exception as e:
                print(f"Error in face_tracker.process: {e}")
                import traceback; traceback.print_exc()
                face_detected = False

            try:
                hand_detected = self.hand_tracker.process(frame)
            except Exception as e:
                print(f"Error in hand_tracker.process: {e}")
                import traceback; traceback.print_exc()
                hand_detected = False

            try:
                hand_landmarks = self.hand_tracker.get_all_landmarks()
                self.cigarette_tracker.update(hand_landmarks)
            except Exception as e:
                print(f"Error in cigarette_tracker.update: {e}")
                import traceback; traceback.print_exc()

            interaction_state = CigaretteMouthState.FAR
            interaction_distance = None
            mouth_center = None
            smoking_state = SmokingState.IDLE
            pattern_detected = False
            exhalation_detected = False

            try:
                if face_detected and self.cigarette_tracker.is_held:
                    mouth_center = self.face_tracker.get_mouth_center()
                    interaction_state = self.cigarette_mouth_detector.update(self.cigarette_tracker, self.face_tracker)
                    interaction_distance = self.cigarette_mouth_detector.get_distance()
                    smoking_state = self.smoking_detector.update(
                        self.cigarette_tracker, self.cigarette_mouth_detector, self.face_tracker
                    )
                    pattern_detected = self.smoking_detector.is_pattern_detected()
                    exhalation_detected = self.smoking_detector.is_exhalation_detected()
            except Exception as e:
                print(f"Error in interaction/smoking detection: {e}")
                import traceback; traceback.print_exc()

            is_inhaling = (smoking_state == SmokingState.INHALING)
            
            # Update glow - for 3D renderer, use built-in glow; for 2D, use separate glow effect
            if self.use_3d and self.cigarette_renderer_3d:
                self.cigarette_renderer_3d.update_glow(is_inhaling)
            else:
                self.glow_effect.set_target(is_inhaling)
                self.glow_effect.update()

            self.smoke_effect.update(exhalation_detected, mouth_center)

            # Render AR effects
            if self.cigarette_tracker.is_held and self.cigarette_tracker.position is not None:
                if self.use_3d and self.cigarette_renderer_3d:
                    # Use 3D renderer (handles glow internally)
                    frame = self.cigarette_renderer_3d.render(frame, self.cigarette_tracker)
                else:
                    # 2D fallback
                    if self.cigarette_renderer.cigarette_img is not None:
                        self.cigarette_renderer.draw(frame, self.cigarette_tracker.position, self.cigarette_tracker.rotation, 0.0)
                    else:
                        self.fallback_renderer.draw(frame, self.cigarette_tracker.position, self.cigarette_tracker.rotation, 0.0)

                    # Apply 2D glow effect
                    self.glow_effect.draw(frame, self.cigarette_tracker.position, self.cigarette_tracker.rotation, self.cigarette_tracker.length)

            self.smoke_effect.draw(frame)

            # Debug overlay
            if self.debug_mode:
                self._draw_debug_overlay(frame, face_detected, hand_detected, interaction_state, interaction_distance, mouth_center, smoking_state, pattern_detected, exhalation_detected, is_inhaling)

            cv2.imshow('Virtual Smoking', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('D') or key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('3'):
                if self.cigarette_renderer_3d:
                    self.use_3d = not self.use_3d
                    print(f"3D Renderer: {'ON' if self.use_3d else 'OFF (2D fallback)'}")
                else:
                    print("3D renderer is unavailable; continuing in 2D mode")
            elif self.debug_mode:
                pass  # Debug keys only work in debug mode

        self.shutdown()

    def _draw_debug_overlay(self, frame, face_detected, hand_detected, interaction_state, interaction_distance, mouth_center, smoking_state, pattern_detected, exhalation_detected, is_inhaling):
        """Draw debug information overlay."""
        if self.debug_mode:
            self.face_tracker.draw_landmarks(frame)
            self.hand_tracker.draw_landmarks(frame)

        if self.debug_mode and mouth_center and self.cigarette_tracker.is_held:
            cig_pos = self.cigarette_tracker.get_mouth_end_position(mouth_center)
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
        cv2.putText(frame, f"Cigarette: {'HELD' if self.cigarette_tracker.is_held else 'NOT HELD'}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 0) if self.cigarette_tracker.is_held else (0, 0, 255), 2)
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

        cv2.putText(frame, f"FPS: {self.camera.get_fps():.1f}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.debug_mode:
            debug = self.cigarette_mouth_detector.get_debug_info()
            status_y += 30
            cv2.putText(frame, f"Threshold: {debug['threshold']} px",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25
            cv2.putText(frame, f"Frames in state: {debug['frames_in_state']}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25
            cv2.putText(frame, f"Approach: {debug['approach_count']} Near: {debug['near_count']} Away: {debug['away_count']}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            smoking_debug = self.smoking_detector.get_debug_info()
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

            status_y += 30
            cv2.putText(frame, f"Glow Intensity: {self.glow_effect.get_intensity():.2f}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_y += 25
            cv2.putText(frame, f"Glow Target: {'ON' if is_inhaling else 'OFF'}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25
            cv2.putText(frame, f"Fade In: {self.glow_effect.fade_in_speed} Fade Out: {self.glow_effect.fade_out_speed}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            status_y += 30
            cv2.putText(frame, f"Smoke: {'ACTIVE' if self.smoke_effect.is_active() else 'INACTIVE'}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            status_y += 25
            cv2.putText(frame, f"Particles: {self.smoke_effect.get_particle_count()}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            status_y += 25
            cv2.putText(frame, f"Exhalation Triggered: {'YES' if self.smoke_effect.exhalation_triggered else 'NO'}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if self.cigarette_tracker.is_held:
                cig_debug = self.cigarette_tracker.get_debug_info()
                status_y += 30
                cv2.putText(frame, f"Cig Pos: ({cig_debug['position'][0]:.0f}, {cig_debug['position'][1]:.0f})" if cig_debug['position'] else "Cig Pos: N/A",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                status_y += 25
                cv2.putText(frame, f"Cig Angle: {cig_debug['rotation_deg']:.1f} deg",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if face_detected:
                mouth_data = self.face_tracker.get_mouth_measurements()
                status_y += 30
                cv2.putText(frame, f"Mouth W: {mouth_data['width']:.1f} H: {mouth_data['height']:.1f} Open: {mouth_data['opening']:.1f} AR: {mouth_data['aspect_ratio']:.2f}",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            status_y += 30
            cv2.putText(frame, "Press 'q'/'ESC' to quit, 'D' to toggle debug",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def shutdown(self):
        """Clean shutdown of all resources."""
        self.running = False
        if self.camera:
            self.camera.close()
            self.camera = None
        if self.face_tracker:
            self.face_tracker.close()
            self.face_tracker = None
        if self.hand_tracker:
            self.hand_tracker.close()
            self.hand_tracker = None
        if self.cigarette_renderer_3d:
            self.cigarette_renderer_3d.close()
            self.cigarette_renderer_3d = None
        cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run the Virtual Smoking AR application')
    parser.add_argument('-d', '--debug', action='store_true', help='show tracking and state overlays')
    parser.add_argument('--2d', dest='prefer_3d', action='store_false', help='use the 2D renderer')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    app = VirtualSmokingApp(debug_mode=args.debug, prefer_3d=args.prefer_3d)
    if not app.initialize():
        return 1

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Runtime error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        app.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
