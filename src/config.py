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
        'processing_scale': 0.55,
    }

    HAND_TRACKER = {
        'max_hands': 2,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'processing_scale': 0.72,
    }

    CIGARETTE_TRACKER = {
        'length': 140,
        'thickness': 12,
        'reference_hand_length': 190,
        'min_length': 85,
        'max_length': 210,
        'grip_tip_ratio': 0.55,
        'max_grip_distance_ratio': 0.42,
        'release_grip_distance_ratio': 0.62,
        'grab_radius': 140,
        # Offset the model center outward so the fingers grip the filter area.
        'grip_to_center_ratio': 0.30,
        'tracking_loss_velocity_decay': 0.65,
        'gravity': 0.85,
        'air_drag': 0.992,
        'angular_drag': 0.985,
        'max_release_speed': 28,
        'max_angular_velocity': 0.25,
        'respawn_margin': 80,
        'default_depth_rotation': np.deg2rad(70.0),
        'max_depth_rotation': np.deg2rad(82.0),
        'position_smoothing': {
            'freq': 30.0,
            'mincutoff': 2.5,
            'beta': 0.70,
        },
        'rotation_smoothing': {
            'freq': 30.0,
            'mincutoff': 1.8,
            'beta': 0.35,
        },
        'depth_smoothing': {
            'freq': 30.0,
            'mincutoff': 1.2,
            'beta': 0.18,
        },
        'max_frames_lost': 5,
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
        'exhalation_duration_frames': 45,
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

    CIGARETTE_3D = {
        'model_scale': 1.0,
        'model_offset_x': 0.0,
        'model_offset_y': 0.0,
        'model_offset_z': 0.0,
        'model_rotation_offset_x': 0.0,
        # Live depth tilt comes from the 3D hand pose.
        'model_rotation_offset_y': 0.0,
        'model_rotation_offset_z': 0.0,
    }

    SMOKE_EFFECT = {
        # Emit a few particles continuously instead of one large, abrupt puff.
        'particle_count_min': 2,
        'particle_count_max': 4,
        'spawn_interval_frames': 2,
        'max_particles': 100,
        'blur_sigma': 4.5,
        'initial_size_min': 5,
        'initial_size_max': 10,
        'initial_opacity_min': 0.08,
        'initial_opacity_max': 0.17,
        'lifetime_min': 55,
        'lifetime_max': 90,
        'min_speed': 0.8,
        'max_speed': 1.6,
        'direction_screen_gain': 2.0,
        'direction_spread': 0.65,
        'direction_follow_frames': 18,
        'direction_follow_strength': 0.38,
        'forward_spread_min': 0.32,
        'forward_spread_max': 0.75,
        'ambient_drift_x_min': -0.28,
        'ambient_drift_x_max': 0.28,
        'ambient_drift_y_min': -0.16,
        'ambient_drift_y_max': -0.04,
        'velocity_drag': 0.985,
        'upward_force': 0.028,
        'turbulence_strength_min': 0.04,
        'turbulence_strength_max': 0.11,
        'turbulence_frequency_min': 0.08,
        'turbulence_frequency_max': 0.16,
        'expansion_rate_min': 0.35,
        'expansion_rate_max': 0.7,
        'aspect_ratio_min': 0.72,
        'aspect_ratio_max': 1.4,
        'density_texture_strength': 0.22,
        'color_r_min': 195,
        'color_r_max': 225,
        'color_g_min': 195,
        'color_g_max': 225,
        'color_b_min': 198,
        'color_b_max': 228,
        'origin_offset_x': 0,
        'origin_offset_y': -2,
    }

    INHALE_SMOKE_EFFECT = {
        'particle_count_min': 1,
        'particle_count_max': 2,
        'spawn_interval_frames': 3,
        'travel_frames_min': 12,
        'travel_frames_max': 20,
        'initial_size_min': 3,
        'initial_size_max': 6,
        'initial_opacity_min': 0.18,
        'initial_opacity_max': 0.35,
        'spawn_jitter': 3,
        'shrink_rate': 0.05,
        'fade_rate': 0.07,
        'curve_strength_min': 4,
        'curve_strength_max': 10,
    }

    DEBUG = {
        'show_fps': True,
        'font_scale': 0.7,
        'font_thickness': 2,
        'line_spacing': 30,
    }
