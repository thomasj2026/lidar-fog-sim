"""Scene definition of the model (map) """
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml

from .model import Wall


def box_walls(center, size, yaw, rho):
    """Four walls of a rectangular obstacle, normals pointing out of the box."""
    hx, hy = size[0] / 2.0, size[1] / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])      # CCW
    pts = (R @ corners.T).T + np.asarray(center, float)
    pts = pts[::-1]                                                        # CW
    return [Wall(pts[i], pts[(i + 1) % 4], rho) for i in range(4)]


@dataclass
class Scene:
    name: str
    room_walls: list                      # walls from the 'walls' list
    boxes: list                           # raw dicts from the 'boxes' list
    walls: list = field(default_factory=list)   # model map: room walls + expanded boxes

    def is_free(self, p, margin: float = 0.3) -> bool:
        """True if p is on the free side of every room wall and outside every (inflated) box."""
        p = np.asarray(p, float)
        if any(w.n @ p - w.d < margin for w in self.room_walls):
            return False
        for b in self.boxes:
            c, s = np.cos(-b.get("yaw", 0.0)), np.sin(-b.get("yaw", 0.0))
            q = np.array([[c, -s], [s, c]]) @ (p - np.asarray(b["center"], float))
            if abs(q[0]) < b["size"][0] / 2 + margin and abs(q[1]) < b["size"][1] / 2 + margin:
                return False
        return True


def load_scene(path: str, name: str) -> Scene:
    with open(path) as f:
        cfg = yaml.safe_load(f)["scenes"][name]
    room = [Wall(w["a"], w["b"], w.get("rho", 0.8)) for w in cfg.get("walls", [])]
    boxes = cfg.get("boxes", []) or []
    walls = list(room)
    for b in boxes:
        walls += box_walls(b["center"], b["size"], b.get("yaw", 0.0), b.get("rho", 0.8))
    return Scene(name, room, boxes, walls)
