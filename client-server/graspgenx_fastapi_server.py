#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GraspGenX FastAPI inference server.

A thin HTTP wrapper around :class:`GraspGenXSampler`, so a client in *another*
Python environment (e.g. the Isaac-Sim ``env_isaaclab`` running r2s2r's collect
loop) can request 6-DOF grasps without importing torch / graspgenx / CUDA. The
server holds the GPU + model resident and lazily loads (and caches) one sampler
per gripper, mirroring the bundled ZMQ server's semantics.

Run it inside the GraspGenX uv venv:

    .venv/bin/python client-server/graspgenx_fastapi_server.py \
        --config ext/graspgenx_checkpoints/release \
        --assets_dir assets \
        --default_gripper omnipicker \
        --host 0.0.0.0 --port 8754

``--config`` is the checkpoint ROOT containing ``gen/`` and ``dis/`` subdirs
(``load_model_cfg`` merges them). With ``--config`` omitted it auto-resolves the
release checkpoints (``get_checkpoints_version_dir()``), auto-downloading on
first run.

Endpoints:
  GET  /health                         -> {"status": "ok", "default_gripper": ...}
  POST /infer  {points:[[x,y,z]...],   -> {"grasps":[[4x4]...], "confidences":[...],
                gripper_name?, num_grasps?,    "gripper_name": str, "count": int}
                grasp_threshold?, topk_num_grasps?}

``points`` are in the object's own frame (any units the model was trained on —
i.e. real-world metres). Returned ``grasps`` are 4x4 matrices in that same frame
(GraspGenX convention: approach=+Z, contact=±X, origin at gripper base); the
r2s2r client converts them to the pipeline's pose_set convention.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Resolve graspgenx from the repo checkout this file lives in.
_REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graspgenx.grasp_server import GraspGenXSampler  # noqa: E402
from graspgenx.utils.checkpoint_io import load_model_cfg  # noqa: E402


class InferRequest(BaseModel):
    points: list[list[float]]
    gripper_name: Optional[str] = None
    num_grasps: int = 800
    grasp_threshold: float = 0.85
    topk_num_grasps: int = -1
    remove_outliers: bool = False


class _ServerState:
    """Holds the merged config + a per-gripper sampler cache (thread-safe)."""

    def __init__(self, checkpoint_root: str, assets_dir: str, default_gripper: Optional[str]):
        gen_dir = os.path.join(checkpoint_root, "gen")
        dis_dir = os.path.join(checkpoint_root, "dis")
        self.cfg = load_model_cfg(gen_dir, dis_dir)
        self.assets_dir = assets_dir
        self.default_gripper = default_gripper
        self._samplers: dict[str, GraspGenXSampler] = {}
        self._lock = threading.Lock()
        if default_gripper:
            self.get_sampler(default_gripper)

    def get_sampler(self, gripper_name: str) -> GraspGenXSampler:
        with self._lock:
            sampler = self._samplers.get(gripper_name)
            if sampler is None:
                sampler = GraspGenXSampler(self.cfg, gripper_name, assets_dir=self.assets_dir)
                self._samplers[gripper_name] = sampler
            return sampler


_state: _ServerState | None = None
app = FastAPI(title="GraspGenX Inference Server", version="0.1.0")


@app.get("/health")
def health() -> dict:
    assert _state is not None
    return {
        "status": "ok",
        "default_gripper": _state.default_gripper,
        "loaded_grippers": sorted(_state._samplers.keys()),
    }


@app.post("/infer")
def infer(req: InferRequest) -> dict:
    assert _state is not None
    gripper_name = req.gripper_name or _state.default_gripper
    if not gripper_name:
        raise HTTPException(400, "request omitted 'gripper_name' and no --default_gripper is set")

    pts = np.asarray(req.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 16:
        raise HTTPException(400, f"points must be (N>=16, 3); got {pts.shape}")

    try:
        sampler = _state.get_sampler(gripper_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"failed to load gripper '{gripper_name}': {exc}") from exc

    t0 = time.time()
    grasps_t, conf_t = GraspGenXSampler.run_inference(
        pts,
        sampler,
        grasp_threshold=req.grasp_threshold,
        num_grasps=req.num_grasps,
        topk_num_grasps=req.topk_num_grasps,
        remove_outliers=req.remove_outliers,
    )
    grasps = grasps_t.cpu().numpy().astype(float) if len(grasps_t) else np.empty((0, 4, 4))
    conf = conf_t.cpu().numpy().astype(float) if len(conf_t) else np.empty((0,))

    info = sampler.get_gripper_info()
    return {
        "grasps": grasps.tolist(),
        "confidences": conf.tolist(),
        "gripper_name": gripper_name,
        "gripper_depth": float(getattr(info, "depth", 0.0)),
        "gripper_width": float(getattr(info, "width", 0.08)),
        "count": int(len(grasps)),
        "timing": {"infer_ms": round((time.time() - t0) * 1000.0, 1)},
    }


def parse_args():
    p = argparse.ArgumentParser(description="GraspGenX FastAPI inference server")
    p.add_argument("--config", type=str, default=None,
                   help="Checkpoint root with gen/ and dis/ (default: auto-resolve release)")
    p.add_argument("--assets_dir", type=str, default=None,
                   help="Root with x_grippers/ and proc_grippers/ (default: <repo>/assets)")
    p.add_argument("--default_gripper", type=str, default="omnipicker")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8754)
    return p.parse_args()


def main():
    global _state
    args = parse_args()

    checkpoint_root = args.config
    if checkpoint_root is None:
        import graspgenx  # noqa: F401 - import hook auto-downloads checkpoints
        from graspgenx import get_checkpoints_version_dir

        checkpoint_root = str(get_checkpoints_version_dir())
    assets_dir = args.assets_dir or str(_REPO_ROOT / "assets")

    print(f"[graspgenx-server] checkpoints={checkpoint_root} assets={assets_dir} "
          f"gripper={args.default_gripper}")
    _state = _ServerState(checkpoint_root, assets_dir, args.default_gripper)
    print(f"[graspgenx-server] ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
