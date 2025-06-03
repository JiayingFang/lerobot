#!/usr/bin/env python3
"""
LeRobot dataset builders for Human-Shadow (bimanual).

* **generate_train_lerobot.py** - folder-based (unchanged)
* **convert_npz_to_lerobot.py** - batch conversion from TrainingDataSequence
  ``.npz`` files (one episode per file). **New behaviour**:

  • If *any* frame in the sequence violates the camera-velocity
    thresholds, the **entire sequence is skipped**.  This guarantees all
    included episodes are free from excessive ego-motion.

  • Observation keys:
      • ``observation.state`` (20-D EEF)
      • ``observation.cam_lin_vel`` (3-D)
      • ``observation.cam_ang_vel`` (3-D)
      • ``observation.images.frontview``
  • ``action`` remains the 20-D EEF state of the *next* frame.

Example:
--------
```bash
python convert_npz_to_lerobot.py \
    --npz-dir /data/seqs \
    --repo-id my-org/shadow_npz_v2 \
    --lin-thresh 0.8 \
    --ang-thresh 1.2 \
    --push-to-hub
```
A sequence is skipped if *any* frame has |v| > 0.8 m/s **or** |ω| > 1.2 rad/s.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from dataclasses import dataclass

LEROBOT_HOME = Path("/juno/u/jyfang/lerobot")
HF_LEROBOT_HOME = Path("/juno/u/jyfang/lerobot")
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    """First two columns of R flattened into 6-D."""
    return mat[:, :2].reshape(-1)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    return matrix_to_rotation_6d(Rotation.from_quat(quat).as_matrix())

def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    return matrix_to_rotation_6d(Rotation.from_rotvec(rotvec).as_matrix())


def build_eef_vec(pos: np.ndarray, rot6d: np.ndarray, grip: float) -> np.ndarray:
    return np.concatenate([pos, rot6d, [grip]]).astype(np.float32)

@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: int = 4
    image_writer_threads: int = 2
    video_backend: str | None = None

# -----------------------------------------------------------------------------
# Dataset scaffold
# -----------------------------------------------------------------------------

def create_empty_dataset(repo_id: str, camera_res_h: int, camera_res_w: int, fps: int, cfg: DatasetConfig) -> LeRobotDataset:
    names = []
    for arm in ("left", "right"):
        names += [f"{arm}_{k}" for k in ("x", "y", "z")]
        names += [f"{arm}_r6d_{i}" for i in range(6)]
        names.append(f"{arm}_grip")
    cam_names = []
    for cam_t in range(16):
        cam_names.append(f'cam_{cam_t}_x')
        cam_names.append(f'cam_{cam_t}_y')
        cam_names.append(f'cam_{cam_t}_z')
        cam_names.append(f'cam_{cam_t}_r6d_0')
        cam_names.append(f'cam_{cam_t}_r6d_1')
        cam_names.append(f'cam_{cam_t}_r6d_2')
        cam_names.append(f'cam_{cam_t}_r6d_3')
        cam_names.append(f'cam_{cam_t}_r6d_4')
        cam_names.append(f'cam_{cam_t}_r6d_5')

    features = {
        "observation.state": {"dtype": "float32", "shape": (164,), "names": names+cam_names},
        "action": {"dtype": "float32", "shape": (20,), "names": names},
        "observation.images": {
            "dtype": "image", "shape": (3, camera_res_h, camera_res_w), "names": ["C", "H", "W"]
        },
        "observation.language_embedding": {"dtype": "float32", "shape": (768,), "names": ["language"]},
    }

    dst = LEROBOT_HOME / repo_id
    if dst.exists():
        shutil.rmtree(dst)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="human_shadow_bimanual",
        features=features,
        use_videos=cfg.use_videos,
        tolerance_s=cfg.tolerance_s,
        image_writer_processes=cfg.image_writer_processes,
        image_writer_threads=cfg.image_writer_threads,
        video_backend=cfg.video_backend,
    )

# -----------------------------------------------------------------------------
# NPZ → episode (with sequence-level velocity guard)
# -----------------------------------------------------------------------------

def transform_to_current_frame(pos: np.ndarray, rot6d: np.ndarray, current_pos: np.ndarray, current_rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform a position and rotation from world frame to current frame."""
    # Convert 6D rotation to matrix for both current and target
    current_rot_mat = Rotation.from_matrix(np.column_stack([current_rot6d.reshape(2, 3), np.cross(current_rot6d.reshape(2, 3)[0], current_rot6d.reshape(2, 3)[1])])).as_matrix()
    target_rot_mat = Rotation.from_matrix(np.column_stack([rot6d.reshape(2, 3), np.cross(rot6d.reshape(2, 3)[0], rot6d.reshape(2, 3)[1])])).as_matrix()
    
    # Compute relative transformation
    rel_pos = pos - current_pos
    rel_pos = current_rot_mat.T @ rel_pos  # Rotate position into current frame
    rel_rot_mat = current_rot_mat.T @ target_rot_mat
    rel_rot6d = matrix_to_rotation_6d(rel_rot_mat)
    
    return rel_pos, rel_rot6d

def npz_to_episode(path: Path, lang_path: Path, ds: LeRobotDataset, camera_res_h: int, camera_res_w: int, lin_thr: float | None, ang_thr: float | None):
    d = np.load(path, allow_pickle=True)
    lang_emb = np.load(lang_path, allow_pickle=True)

    valid = d["valid"].astype(bool)
    cam_lin = d["cam_lin_vel"]  # (T,3)
    cam_ang = d["cam_ang_vel"]  # (T,3)

    # Check thresholds ---------------------------------------------------------
    if lin_thr is not None and np.linalg.norm(cam_lin, axis=1).max() > lin_thr*10:
        raise ValueError("sequence exceeds linear-velocity threshold")
    if ang_thr is not None and np.linalg.norm(cam_ang, axis=1).max() > ang_thr*10:
        raise ValueError("sequence exceeds angular-velocity threshold")
    print(np.linalg.norm(cam_lin, axis=1).max())
    keep_idx = np.where(valid)[0]
    if len(keep_idx) < 2:
        raise ValueError("<2 valid frames")

    # Extract arrays -----------------------------------------------------------
    pos_l, pos_r = d["action_pos_left"], d["action_pos_right"]
    quat_l, quat_r = d["action_orixyzw_left"], d["action_orixyzw_right"]
    grip_l, grip_r = d["action_gripper_left"].ravel(), d["action_gripper_right"].ravel()
    imgs = d["img_overlay"]

    rot6d_cam = np.stack([rotvec_to_rot6d(rot_vec) for rot_vec in cam_ang])

    # Build episode ------------------------------------------------------------
    usable_pairs = keep_idx[np.isin(keep_idx + 1, keep_idx)]
    for t in usable_pairs:
        # Current frame poses
        rot6d_l = quat_xyzw_to_rot6d(quat_l[t])
        rot6d_r = quat_xyzw_to_rot6d(quat_r[t])
        current_pos_l = pos_l[t]
        current_pos_r = pos_r[t]
        
        # Get sequence of current + 15 future valid camera frames
        valid_after = keep_idx[keep_idx >= t]
        seq_indices = valid_after[:16]
        
        # Initialize arrays with zeros
        cam_lin_seq = np.zeros((16, 3), dtype=np.float32)
        cam_rot_seq = np.zeros((16, 6), dtype=np.float32)
        
        # Fill in the available valid frames
        n_valid = len(seq_indices)
        cam_lin_seq[:n_valid] = cam_lin[seq_indices] / 10
        cam_rot_seq[:n_valid] = rot6d_cam[seq_indices] / 10

        # Get future actions and transform them to current frame
        future_actions = np.zeros((16, 20), dtype=np.float32)  # 16 timesteps, 20-dim actions
        
        for i, future_t in enumerate(seq_indices):
            # Left arm
            future_pos_l = pos_l[future_t]
            future_rot6d_l = quat_xyzw_to_rot6d(quat_l[future_t])
            rel_pos_l, rel_rot6d_l = transform_to_current_frame(future_pos_l, future_rot6d_l, current_pos_l, rot6d_l)
            
            # Right arm
            future_pos_r = pos_r[future_t]
            future_rot6d_r = quat_xyzw_to_rot6d(quat_r[future_t])
            rel_pos_r, rel_rot6d_r = transform_to_current_frame(future_pos_r, future_rot6d_r, current_pos_r, rot6d_r)
            
            # Build action vector in current frame
            future_actions[i] = np.concatenate([
                build_eef_vec(rel_pos_l, rel_rot6d_l, grip_l[future_t]),
                build_eef_vec(rel_pos_r, rel_rot6d_r, grip_r[future_t])
            ])
        
        # Flatten sequences for observation
        cam_lin_flat = cam_lin_seq.reshape(-1)
        cam_rot_flat = cam_rot_seq.reshape(-1)
        
        obs = np.concatenate([
            build_eef_vec(pos_l[t], rot6d_l, grip_l[t]),
            build_eef_vec(pos_r[t], rot6d_r, grip_r[t]),
            cam_lin_flat,
            cam_rot_flat,
        ])
        
        # Next action in world frame (for backward compatibility)
        rot6d_la = quat_xyzw_to_rot6d(quat_l[t+1])
        rot6d_ra = quat_xyzw_to_rot6d(quat_r[t+1])
        act = np.concatenate([
            build_eef_vec(pos_l[t + 1], rot6d_la, grip_l[t + 1]),
            build_eef_vec(pos_r[t + 1], rot6d_ra, grip_r[t + 1]),
        ])
        
        frame = {
            "observation.state": obs,
            "action": act,
            "future_actions": future_actions,  # Add future actions in current frame
            "observation.images": cv2.resize(imgs[t], (camera_res_w, camera_res_h)).transpose(2, 0, 1),
            "observation.language_embedding": lang_emb.astype(np.float32),
            "task": "wild_human_epic_v0"
        }
        ds.add_frame(frame)

    ds.save_episode()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert TrainingDataSequence .npz to LeRobot (skip seq on high cam-vel)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz-dir", help="Directory with .npz files")
    g.add_argument("--npz-files", nargs="+", help="Explicit .npz list")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--camera-res-h", type=int, default=128)
    ap.add_argument("--camera-res-w", type=int, default=228)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--lin-thresh", type=float, default=0.5, help="Skip seq if any |v|>thr; ≤0 disables")
    ap.add_argument("--ang-thresh", type=float, default=0.5, help="Skip seq if any |ω|>thr; ≤0 disables")
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()

    lin_thr = None if args.lin_thresh <= 0 else args.lin_thresh
    ang_thr = None if args.ang_thresh <= 0 else args.ang_thresh

    if args.npz_dir:
        base_path = Path(args.npz_dir)
        folders = ["P01_01", "P01_02", "P01_03", "P01_04", "P01_05", "P01_06", "P01_08", "P01_09", "P01_10", "P01_102", "P01_103", "P01_104", "P01_105"]
            # Only get files from specified folders
        files = []
        for folder in folders:
            folder_path = base_path / folder
            if folder_path.exists():
                for subfolder in folder_path.iterdir():
                    files.extend(subfolder.glob("inpaint_processor/training_data_shoulders.npz")) # TODO: change to *training_data_shoulders.npz
        files = sorted(files)
        print("number of files: ", len(files))
        # else:
            # Get all files as before
            # files = sorted(base_path.glob("*training_data_shoulders.npz"))
    else:
        files = [Path(f) for f in args.npz_files]
    
    assert files, "No .npz files found"

    ds = create_empty_dataset(args.repo_id, args.camera_res_h, args.camera_res_w, args.fps, DatasetConfig())

    kept, skipped = 0, 0
    for f in tqdm.tqdm(files, desc="episodes"):
        try:
            lang_f = Path(str(f).replace("inpaint_processor/training_data_shoulders.npz", "lang_distilbert.npy"))
            npz_to_episode(f, lang_f, ds, args.camera_res_h, args.camera_res_w, lin_thr, ang_thr)
            kept += 1
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}")
            skipped += 1

    print(f"Kept {kept} / {len(files)} episodes; skipped {skipped}.")

    if kept == 0:
        print("No episodes retained - aborting dataset creation.")
        sys.exit(1)

    if args.push_to_hub:
        ds.push_to_hub()

if __name__ == "__main__":
    main()