import os

import cv2
import numpy as np


def save_episode_video(frames, filepath, fps=30):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))
    for frame in frames:
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    out.release()
    print(f"[Video] Saved episode video: {filepath} ({len(frames)} frames)")


def get_raw_frame(env):
    raw_frame = getattr(env, '_last_raw_frame', None)
    if raw_frame is None:
        inner = env
        while hasattr(inner, 'env'):
            if hasattr(inner, '_last_raw_frame'):
                raw_frame = inner._last_raw_frame
                break
            inner = inner.env
        # Check innermost env after loop exits (FPVEnv doesn't have .env attribute)
        if raw_frame is None and hasattr(inner, '_last_raw_frame'):
            raw_frame = inner._last_raw_frame
    return raw_frame


class VideoRecorder:
    def __init__(self, logdir, video_every=5000, enabled=True):
        self.enabled = enabled and video_every > 0
        self.video_every = video_every
        self.last_video_step = 0
        self.current_frames = []
        self.video_dir = None

        if self.enabled:
            self.video_dir = logdir / 'videos'
            os.makedirs(str(self.video_dir), exist_ok=True)

    def should_record(self, step):
        if not self.enabled:
            return False
        return (step - self.last_video_step) >= self.video_every

    def on_step(self, step, env, obs, print_fn=print):
        if not self.should_record(step):
            return False

        # Clear frames on episode start
        if obs.get('is_first', False):
            self.current_frames.clear()

        # Collect frame
        raw_frame = get_raw_frame(env)
        if raw_frame is not None:
            self.current_frames.append(raw_frame.copy())
        elif 'image' in obs:
            # Fallback to observation image
            self.current_frames.append(obs['image'].copy())

        # Save on episode end
        if obs.get('is_last', False) and self.current_frames:
            video_path = self.video_dir / f'episode_{step:08d}.mp4'
            save_episode_video(self.current_frames, video_path, fps=30)
            self.current_frames.clear()
            self.last_video_step = step
            return True

        return False
