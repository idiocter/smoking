import numpy as np


class Config:
    CAMERA = {
        'device_index': 0,
        'width': 1280,
        'height': 720,
        'fps': 30,
    }

    FACE_TRACKER = {
        'max_faces': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'mouth_smoothing_window': 5,
    }

    HAND_TRACKER = {
        'max_hands': 2,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
    }

    CIGARETTE_TRACKER = {
        'length': 140,
        'thickness': 12,
        'position_smoothing': {
            'freq': 30.0,
            'mincutoff': 1.5,
            'beta': 0.3,
        },
        'rotation_smoothing': {
            'freq': 30.0,
            'mincutoff': 1.0,
            'beta': 0.5,
        },
        'max_frames_lost': 10,
        'min_finger_distance': 15,
    }

    CIGARETTE_MOUTH_DETECTOR = {
        'near_threshold': 80,
        'approach_frames': 3,
        'near_frames': 3,
        'away_frames': 3,
        'distance_smoothing': 3,
        'approach_delta_threshold': 2.0,
        'away_delta_threshold': 2.0,
        'near_exit_multiplier': 1.2,
        'far_exit_multiplier': 2.0,
    }

    SMOKING_DETECTOR = {
        'near_mouth_threshold': 80,
        'approaching_frame_count': 3,
        'near_mouth_frame_count': 3,
        'inhalation_window': 10,
        'inhalation_frame_count': 4,
        'away_frame_count': 3,
        'mouth_opening_change_threshold': 4,
        'mouth_aspect_ratio_change_threshold': 0.3,
        'exhalation_window': 15,
        'exhalation_frame_count': 5,
        'exhalation_stability_frames': 2,
        'exhalation_mouth_opening_threshold': 6,
        'exhalation_mouth_width_change_threshold': 4,
        'away_from_mouth_threshold': 120,
    }

    GLOW_EFFECT = {
        'max_intensity': 1.0,
        'fade_in_speed': 0.15,
        'fade_out_speed': 0.08,
    }

    SMOKE_EFFECT = {
        'particle_count_min': 8,
        'particle_count_max': 16,
        'spread_angle': 0.5,
        'base_angle': -np.pi / 2,
        'initial_size_min': 6,
        'initial_size_max': 14,
        'initial_opacity_min': 0.3,
        'initial_opacity_max': 0.5,
        'lifetime_min': 30,
        'lifetime_max': 60,
        'min_speed': 1.0,
        'max_speed': 3.0,
        'upward_force': 0.08,
        'drift_strength': 0.15,
        'expansion_rate_min': 0.15,
        'expansion_rate_max': 0.35,
        'fade_rate_min': 0.015,
        'fade_rate_max': 0.03,
        'color_r_min': 160,
        'color_r_max': 200,
        'color_g_min': 160,
        'color_g_max': 200,
        'color_b_min': 160,
        'color_b_max': 200,
        'origin_offset_x': 0,
        'origin_offset_y': -8,
    }

    DEBUG = {
        'show_fps': True,
        'font_scale': 0.7,
        'font_thickness': 2,
        'line_spacing': 30,
    }