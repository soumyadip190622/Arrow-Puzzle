#!/usr/bin/env python3
"""
Arrow Puzzle  -  a relaxing logic game written with pygame.

Goal
----
Every arrow is a snake-like path with an arrow-head at one end.  Tap an arrow
and it slides out of the board in the direction its head points - but only if
nothing is standing in its way.  Tap a blocked arrow and you lose a heart.
Clear the whole board to finish the level.

Controls
--------
  Mouse / touch : tap an arrow
  H             : use a hint
  G             : toggle the dot grid
  R             : restart the level
  ESC           : quit

Requirements:  pip install pygame
Run         :  python arrow_puzzle.py

Architecture
------------
  LevelGenerator  - builds a random, *always solvable* level
  Arrow           - data model (cells, direction)
  Game            - state, input, rules, rendering
Progress (current level) is stored in  ~/.arrow_puzzle_save.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pygame

Cell = Tuple[int, int]
Vec = Tuple[int, int]

DIRECTIONS: List[Vec] = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SAVE_FILE = Path.home() / ".arrow_puzzle_save.json"

# --------------------------------------------------------------------------- #
#  Look & feel
# --------------------------------------------------------------------------- #
BG = (238, 243, 250)
PANEL = (255, 255, 255)
SHADOW = (214, 222, 236)
INK = (26, 31, 78)          # arrow colour (dark navy like the original)
INK_SOFT = (120, 126, 160)
DOT = (196, 203, 220)
BLUE = (30, 144, 255)       # hint colour
RED = (232, 64, 64)
HEART_ON = (235, 52, 62)
HEART_OFF = (205, 210, 224)
GREEN = (38, 190, 120)
WHITE = (255, 255, 255)

FPS = 60
START_LIVES = 3


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
@dataclass
class Arrow:
    id: int
    cells: List[Cell]        # ordered tail -> head
    direction: Vec           # direction the head points to

    @property
    def head(self) -> Cell:
        return self.cells[-1]


def ray_cells(head: Cell, direction: Vec, cols: int, rows: int) -> Iterator[Cell]:
    """Cells from just in front of `head` to the edge of the board."""
    c, r = head[0] + direction[0], head[1] + direction[1]
    while 0 <= c < cols and 0 <= r < rows:
        yield (c, r)
        c += direction[0]
        r += direction[1]


# --------------------------------------------------------------------------- #
#  Level generation
# --------------------------------------------------------------------------- #
class LevelGenerator:
    """
    Generates a level that is guaranteed to be solvable.

    Idea: arrows are placed one after another.  A new arrow may only be placed
    if the straight line in front of its head is free of every arrow placed
    *before* it.  Therefore, removing the arrows in the reverse order of
    placement is always a valid solution.  Arrows are placed from the centre
    outwards, which produces many interesting dependencies (inner arrows have
    to wait for outer arrows to leave).
    """

    def __init__(self, cols: int, rows: int, max_len: int, rng: random.Random):
        self.cols, self.rows, self.max_len, self.rng = cols, rows, max_len, rng

    def generate(self) -> List[Arrow]:
        cols, rows, rng = self.cols, self.rows, self.rng
        cx, cy = (cols - 1) / 2, (rows - 1) / 2
        cells = [(c, r) for c in range(cols) for r in range(rows)]
        cells.sort(key=lambda p: math.hypot(p[0] - cx, p[1] - cy) + rng.uniform(0, 2.2))

        occupied: Dict[Cell, int] = {}
        arrows: List[Arrow] = []
        for start in cells:
            if start in occupied:
                continue
            arrow = self._try_place(start, occupied, len(arrows))
            if arrow:
                arrows.append(arrow)
                for cell in arrow.cells:
                    occupied[cell] = arrow.id
        return arrows

    # -- helpers ---------------------------------------------------------- #
    def _try_place(self, start: Cell, occupied: Dict[Cell, int], arrow_id: int) -> Optional[Arrow]:
        rng = self.rng
        for _ in range(14):
            target = rng.randint(2, self.max_len) if self.max_len >= 2 else 1
            path = self._random_walk(start, target, occupied)
            for cells in (path, path[::-1]):
                direction = self._head_direction(cells, occupied)
                if direction:
                    return Arrow(arrow_id, list(cells), direction)

        # Fallback: a single-cell arrow
        dirs = DIRECTIONS[:]
        rng.shuffle(dirs)
        for d in dirs:
            if self._ray_is_free(start, d, {start}, occupied):
                return Arrow(arrow_id, [start], d)
        return None

    def _random_walk(self, start: Cell, length: int, occupied: Dict[Cell, int]) -> List[Cell]:
        rng = self.rng
        path = [start]
        seen = {start}
        last_dir: Optional[Vec] = None
        while len(path) < length:
            c, r = path[-1]
            options: List[Vec] = []
            for d in DIRECTIONS:
                n = (c + d[0], r + d[1])
                if 0 <= n[0] < self.cols and 0 <= n[1] < self.rows \
                        and n not in occupied and n not in seen:
                    options.append(d)
            if not options:
                break
            # prefer going straight - gives nicer, longer corridors
            if last_dir in options and rng.random() < 0.55:
                d = last_dir
            else:
                d = rng.choice(options)
            last_dir = d
            nxt = (c + d[0], r + d[1])
            path.append(nxt)
            seen.add(nxt)
        return path

    def _head_direction(self, cells: List[Cell], occupied: Dict[Cell, int]) -> Optional[Vec]:
        own = set(cells)
        if len(cells) >= 2:
            d = (cells[-1][0] - cells[-2][0], cells[-1][1] - cells[-2][1])
            return d if self._ray_is_free(cells[-1], d, own, occupied) else None
        dirs = DIRECTIONS[:]
        self.rng.shuffle(dirs)
        for d in dirs:
            if self._ray_is_free(cells[-1], d, own, occupied):
                return d
        return None

    def _ray_is_free(self, head: Cell, d: Vec, own: set, occupied: Dict[Cell, int]) -> bool:
        return all(c not in occupied and c not in own
                   for c in ray_cells(head, d, self.cols, self.rows))


def level_config(level: int) -> dict:
    """Difficulty curve: bigger boards, longer arrows, fewer hints."""
    hard = level % 4 == 0
    cols = min(5 + (level - 1) // 2 + (1 if hard else 0), 13)
    rows = min(int(cols * 1.4), 18)
    return {
        "cols": cols,
        "rows": rows,
        "max_len": min(3 + level // 3, 14),
        "hard": hard,
        "hints": 1 if hard else 2,
    }


def build_level(level: int) -> Tuple[dict, List[Arrow]]:
    cfg = level_config(level)
    min_arrows = max(4, cfg["cols"] * cfg["rows"] // 9)
    best: List[Arrow] = []
    for attempt in range(30):
        rng = random.Random(level * 1009 + 17 + attempt * 7919)   # deterministic per level
        arrows = LevelGenerator(cfg["cols"], cfg["rows"], cfg["max_len"], rng).generate()
        if len(arrows) > len(best):
            best = arrows
        if len(arrows) >= min_arrows:
            return cfg, arrows
    return cfg, best


# --------------------------------------------------------------------------- #
#  Geometry / drawing helpers
# --------------------------------------------------------------------------- #
def polyline_length(pts: List[Tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def slice_polyline(pts, start: float, end: float):
    """Return the part of a polyline between two distances along it."""
    out = []
    dist = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        if seg == 0:
            continue
        s0, s1 = dist, dist + seg
        if s1 >= start and s0 <= end:
            t0, t1 = max(start, s0), min(end, s1)
            p0 = (a[0] + (b[0] - a[0]) * (t0 - s0) / seg, a[1] + (b[1] - a[1]) * (t0 - s0) / seg)
            p1 = (a[0] + (b[0] - a[0]) * (t1 - s0) / seg, a[1] + (b[1] - a[1]) * (t1 - s0) / seg)
            if not out:
                out.append(p0)
            out.append(p1)
        dist = s1
    return out


def draw_arrow_shape(surf, pts, direction: Vec, color, cs: float):
    """Draw a thick rounded polyline with an arrow-head at its last point."""
    if len(pts) < 2:
        return
    w = max(3, int(cs * 0.15))
    pygame.draw.lines(surf, color, False, pts, w)
    for p in pts:                                   # round joints / caps
        pygame.draw.circle(surf, color, (int(p[0]), int(p[1])), w // 2)
    hx, hy = pts[-1]
    dx, dy = direction
    px, py = -dy, dx                                # perpendicular
    tip = (hx + dx * cs * 0.42, hy + dy * cs * 0.42)
    base = (hx - dx * cs * 0.02, hy - dy * cs * 0.02)
    half = cs * 0.27
    pygame.draw.polygon(surf, color, [tip, (base[0] + px * half, base[1] + py * half),
                                      (base[0] - px * half, base[1] - py * half)])


def draw_heart(surf, cx, cy, size, color):
    r = size * 0.28
    pygame.draw.circle(surf, color, (cx - r, cy - r * 0.55), r)
    pygame.draw.circle(surf, color, (cx + r, cy - r * 0.55), r)
    pygame.draw.polygon(surf, color, [(cx - size * 0.54, cy - r * 0.2), (cx + size * 0.54, cy - r * 0.2),
                                      (cx, cy + size * 0.55)])


class Slide:
    """Animation of an arrow gliding out of the board."""

    def __init__(self, pts, direction: Vec, body_len: float, travel: float):
        self.pts, self.direction = pts, direction
        self.body_len, self.travel = body_len, travel
        self.s, self.t = 0.0, 0.0

    def update(self, dt: float):
        self.t += dt
        self.s += (420 + 2600 * self.t) * dt         # accelerate

    @property
    def done(self) -> bool:
        return self.s >= self.travel


# --------------------------------------------------------------------------- #
#  Game
# --------------------------------------------------------------------------- #
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Arrow Puzzle")
        self.screen = pygame.display.set_mode((480, 820), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self._fonts: Dict[Tuple[int, bool], pygame.font.Font] = {}
        self.show_grid = True
        self.level = self._load_progress()
        self.load_level(self.level)

    # -- persistence ------------------------------------------------------- #
    @staticmethod
    def _load_progress() -> int:
        try:
            return max(1, int(json.loads(SAVE_FILE.read_text()).get("level", 1)))
        except Exception:
            return 1

    def _save_progress(self, level: Optional[int] = None):
        try:
            SAVE_FILE.write_text(json.dumps({"level": level or self.level}))
        except OSError:
            pass

    # -- level lifecycle --------------------------------------------------- #
    def load_level(self, level: int):
        self.level = level
        self.cfg, arrows = build_level(level)
        self.cols, self.rows = self.cfg["cols"], self.cfg["rows"]
        self.active: Dict[int, Arrow] = {a.id: a for a in arrows}
        self.owner: Dict[Cell, int] = {c: a.id for a in arrows for c in a.cells}
        self.total = len(arrows)
        self.lives = START_LIVES
        self.hints_left = self.cfg["hints"]
        self.hint_id: Optional[int] = None
        self.slides: List[Slide] = []
        self.shakes: Dict[int, float] = {}
        self.state = "play"             # play | won | lost
        self.state_timer = 0.0
        self.dialog_btn = pygame.Rect(0, 0, 0, 0)
        self._save_progress()

    # -- rules ------------------------------------------------------------- #
    def is_free(self, arrow: Arrow) -> bool:
        return all(c not in self.owner for c in ray_cells(arrow.head, arrow.direction, self.cols, self.rows))

    def tap(self, cell: Cell):
        aid = self.owner.get(cell)
        if aid is None:
            return
        arrow = self.active[aid]
        if self.is_free(arrow):
            self.remove_arrow(arrow)
        else:
            self.lives -= 1
            self.shakes[aid] = 0.45
            if self.hint_id == aid:
                self.hint_id = None
            if self.lives <= 0:
                self.state, self.state_timer = "lost", 0.0

    def remove_arrow(self, arrow: Arrow):
        pts = self.arrow_points(arrow)
        body = polyline_length(pts)
        steps = len(list(ray_cells(arrow.head, arrow.direction, self.cols, self.rows)))
        edge = (steps + 0.5) * self.cs
        hx, hy = pts[-1]
        d = arrow.direction
        far = edge + body + self.cs * 2
        pts.append((hx + d[0] * far, hy + d[1] * far))
        self.slides.append(Slide(pts, d, body, body + edge + self.cs * 0.6))
        for c in arrow.cells:
            self.owner.pop(c, None)
        del self.active[arrow.id]
        if self.hint_id == arrow.id:
            self.hint_id = None

    def use_hint(self):
        if self.state != "play" or self.hints_left <= 0 or not self.active:
            return
        free = [a for a in self.active.values() if self.is_free(a)]
        if free:
            self.hint_id = random.choice(free).id
            self.hints_left -= 1

    # -- layout ------------------------------------------------------------ #
    def layout(self):
        W, H = self.screen.get_size()
        self.W, self.H = W, H
        top, bottom, margin = 120, 120, 26
        avail = pygame.Rect(margin, top, W - 2 * margin, H - top - bottom)
        self.cs = max(8, int(min(avail.w / self.cols, avail.h / self.rows)))
        bw, bh = self.cs * self.cols, self.cs * self.rows
        self.board = pygame.Rect(0, 0, bw, bh)
        self.board.center = avail.center
        by = H - bottom // 2 - 6
        self.hint_btn = pygame.Rect(0, 0, 62, 62)
        self.hint_btn.center = (W // 2 - 50, by)
        self.grid_btn = pygame.Rect(0, 0, 62, 62)
        self.grid_btn.center = (W // 2 + 50, by)

    def cell_center(self, c: int, r: int) -> Tuple[float, float]:
        return (self.board.x + (c + 0.5) * self.cs, self.board.y + (r + 0.5) * self.cs)

    def arrow_points(self, arrow: Arrow):
        pts = [self.cell_center(c, r) for c, r in arrow.cells]
        if len(pts) == 1:                                   # short stub for 1-cell arrows
            x, y = pts[0]
            d = arrow.direction
            pts.insert(0, (x - d[0] * self.cs * 0.3, y - d[1] * self.cs * 0.3))
        return pts

    def pixel_to_cell(self, pos) -> Optional[Cell]:
        if not self.board.collidepoint(pos):
            return None
        return ((pos[0] - self.board.x) // self.cs, (pos[1] - self.board.y) // self.cs)

    # -- text -------------------------------------------------------------- #
    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("segoeui,helveticaneue,arial,dejavusans", size, bold=bold)
        return self._fonts[key]

    def text(self, msg, size, color, center=None, topleft=None, bold=False):
        img = self.font(size, bold).render(str(msg), True, color)
        rect = img.get_rect()
        if center:
            rect.center = center
        if topleft:
            rect.topleft = topleft
        self.screen.blit(img, rect)
        return rect

    # -- update ------------------------------------------------------------ #
    def update(self, dt: float):
        for s in self.slides:
            s.update(dt)
        self.slides = [s for s in self.slides if not s.done]
        for k in list(self.shakes):
            self.shakes[k] -= dt
            if self.shakes[k] <= 0:
                del self.shakes[k]
        if self.state == "play" and not self.active and not self.slides:
            self.state, self.state_timer = "won", 0.0
            self._save_progress(self.level + 1)
        self.state_timer += dt

    # -- drawing ----------------------------------------------------------- #
    def draw(self):
        self.layout()
        scr = self.screen
        scr.fill(BG)
        self.draw_header()

        # board panel
        pygame.draw.rect(scr, SHADOW, self.board.inflate(30, 30).move(0, 4), border_radius=22)
        pygame.draw.rect(scr, PANEL, self.board.inflate(30, 30), border_radius=22)

        # dots on cells without an arrow
        if self.show_grid:
            for c in range(self.cols):
                for r in range(self.rows):
                    if (c, r) not in self.owner:
                        x, y = self.cell_center(c, r)
                        pygame.draw.circle(scr, DOT, (int(x), int(y)), max(2, self.cs // 12))

        # arrows
        ticks = pygame.time.get_ticks() / 1000
        for arrow in self.active.values():
            pts = self.arrow_points(arrow)
            color = INK
            if arrow.id == self.hint_id:
                pulse = 0.5 + 0.5 * math.sin(ticks * 7)
                color = tuple(int(INK[i] + (BLUE[i] - INK[i]) * (0.55 + 0.45 * pulse)) for i in range(3))
            if arrow.id in self.shakes:
                t = self.shakes[arrow.id]
                off = math.sin(t * 60) * self.cs * 0.09
                pts = [(x + off * (1 - abs(arrow.direction[0])), y + off * (1 - abs(arrow.direction[1])))
                       for x, y in pts]
                color = RED
            draw_arrow_shape(scr, pts, arrow.direction, color, self.cs)

        # sliding arrows (clipped to the screen below the header)
        scr.set_clip(pygame.Rect(0, 100, self.W, self.H - 200))
        for s in self.slides:
            part = slice_polyline(s.pts, s.s, s.s + s.body_len)
            draw_arrow_shape(scr, part, s.direction, INK, self.cs)
        scr.set_clip(None)

        self.draw_footer()
        if self.state != "play":
            self.draw_dialog()
        pygame.display.flip()

    def draw_header(self):
        scr = self.screen
        self.text(f"Level {self.level}", 26, INK, center=(self.W // 2, 38), bold=True)

        # remaining arrows (left)
        pill = pygame.Rect(18, 68, 92, 34)
        pygame.draw.rect(scr, WHITE, pill, border_radius=17)
        self.text(f"\u279c {len(self.active) + len(self.slides)}", 17, INK, center=pill.center, bold=True)

        # hearts (centre)
        for i in range(START_LIVES):
            draw_heart(scr, self.W // 2 + (i - 1) * 42, 84, 26, HEART_ON if i < self.lives else HEART_OFF)

        # difficulty (right)
        pill = pygame.Rect(self.W - 110, 68, 92, 34)
        pygame.draw.rect(scr, WHITE, pill, border_radius=17)
        self.text("Hard" if self.cfg["hard"] else "Normal", 16, RED if self.cfg["hard"] else GREEN,
                  center=pill.center, bold=True)

    def draw_footer(self):
        scr = self.screen
        # hint button
        active = self.hints_left > 0 and self.state == "play"
        pygame.draw.circle(scr, WHITE, self.hint_btn.center, 31)
        pygame.draw.circle(scr, BLUE if active else HEART_OFF, self.hint_btn.center, 31, 3)
        self.text("?", 30, BLUE if active else HEART_OFF, center=self.hint_btn.center, bold=True)
        badge = (self.hint_btn.right - 6, self.hint_btn.top + 6)
        pygame.draw.circle(scr, BLUE if active else HEART_OFF, badge, 11)
        self.text(self.hints_left, 14, WHITE, center=badge, bold=True)

        # grid button
        pygame.draw.circle(scr, WHITE, self.grid_btn.center, 31)
        pygame.draw.circle(scr, BLUE if self.show_grid else HEART_OFF, self.grid_btn.center, 31, 3)
        self.text("#", 28, BLUE if self.show_grid else HEART_OFF, center=self.grid_btn.center, bold=True)

        self.text("H: hint   G: grid   R: restart", 14, INK_SOFT, center=(self.W // 2, self.H - 16))

    def draw_dialog(self):
        if self.state_timer < 0.25 and self.state == "lost":
            return                                      # let the shake play first
        veil = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        veil.fill((20, 24, 60, 150))
        self.screen.blit(veil, (0, 0))
        card = pygame.Rect(0, 0, min(360, self.W - 40), 260)
        card.center = (self.W // 2, self.H // 2)
        pygame.draw.rect(self.screen, WHITE, card, border_radius=24)

        won = self.state == "won"
        self.text("Level Complete!" if won else "Out of hearts", 30, GREEN if won else RED,
                  center=(card.centerx, card.y + 60), bold=True)
        self.text("Nice and calm. Ready for the next one?" if won else "Don't worry - try again.",
                  16, INK_SOFT, center=(card.centerx, card.y + 108))
        self.dialog_btn = pygame.Rect(0, 0, 200, 54)
        self.dialog_btn.center = (card.centerx, card.bottom - 62)
        pygame.draw.rect(self.screen, GREEN if won else BLUE, self.dialog_btn, border_radius=27)
        self.text("Next Level" if won else "Retry", 22, WHITE, center=self.dialog_btn.center, bold=True)

    # -- input ------------------------------------------------------------- #
    def on_click(self, pos):
        if self.state != "play":
            if self.dialog_btn.collidepoint(pos):
                self.load_level(self.level + 1 if self.state == "won" else self.level)
            return
        if self.hint_btn.collidepoint(pos):
            self.use_hint()
        elif self.grid_btn.collidepoint(pos):
            self.show_grid = not self.show_grid
        else:
            cell = self.pixel_to_cell(pos)
            if cell:
                self.tap(cell)

    def run(self):
        while True:
            dt = min(self.clock.tick(FPS) / 1000, 0.05)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif e.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((max(360, e.w), max(600, e.h)), pygame.RESIZABLE)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.layout()
                    self.on_click(e.pos)
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif e.key == pygame.K_h:
                        self.use_hint()
                    elif e.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif e.key == pygame.K_r:
                        self.load_level(self.level)
                    elif e.key == pygame.K_RETURN and self.state != "play":
                        self.load_level(self.level + 1 if self.state == "won" else self.level)
            self.layout()
            self.update(dt)
            self.draw()


def main():
    Game().run()


if __name__ == "__main__":
    main()