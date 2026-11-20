"""Grid navigation + line-of-sight built from the static map."""
import array
import heapq
import json
import math
import os
import sys
import time

CELL = 3.0
SQ2 = 1.41421356
DIRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


def _inside(px, py, ring):
    c = False
    n = len(ring) // 2
    j = n - 1
    for i in range(n):
        xi, yi = ring[2 * i], ring[2 * i + 1]
        xj, yj = ring[2 * j], ring[2 * j + 1]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            c = not c
        j = i
    return c


class Nav:
    def __init__(self, mapdir, mech):
        self.ok = False
        self.maxstep = mech.get("step_up", 1.2)
        try:
            self._load(mapdir)
            self.ok = True
        except Exception as e:  # map unreadable: bot still runs without nav
            print("nav load failed:", repr(e), file=sys.stderr)
            self.nx = self.ny = 0

    # ---------- loading ----------
    def _load(self, mapdir):
        w = json.load(open(os.path.join(mapdir, "v2", "world.json")))
        t = w["terrain"]
        self.tminx, self.tminy, self.tstep = t["minX"], t["minY"], t["step"]
        self.tnx, self.tny = t["nx"], t["ny"]
        a = array.array("h")
        with open(os.path.join(mapdir, "v2", t["file"]), "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder != "little":
            a.byteswap()
        self.terr = a
        b = w["bounds"]
        self.minx, self.miny = b["minX"], b["minY"]
        self.nx = int((b["maxX"] - b["minX"]) / CELL) + 1
        self.ny = int((b["maxY"] - b["minY"]) / CELL) + 1
        nx, ny = self.nx, self.ny
        N = nx * ny
        raw = bytearray(N)
        top = [-1e9] * N
        maxslope = math.tan(math.radians(w["rules"].get("maxSlopeDeg", 50)))
        # slope
        for j in range(ny):
            y = self.miny + (j + 0.5) * CELL
            for i in range(nx):
                x = self.minx + (i + 0.5) * CELL
                g = self.ground(x, y)
                gx = (self.ground(x + 1.5, y) - self.ground(x - 1.5, y)) / 3.0
                gy = (self.ground(x, y + 1.5) - self.ground(x, y - 1.5)) / 3.0
                if math.hypot(gx, gy) > maxslope:
                    raw[j * nx + i] = 1
        # prisms
        for p in w["prisms"]:
            ring = p["ring"]
            xs = ring[0::2]
            ys = ring[1::2]
            i0 = max(0, int((min(xs) - self.minx) / CELL))
            i1 = min(nx - 1, int((max(xs) - self.minx) / CELL))
            j0 = max(0, int((min(ys) - self.miny) / CELL))
            j1 = min(ny - 1, int((max(ys) - self.miny) / CELL))
            holes = p.get("holes") or []
            if holes and isinstance(holes[0], (int, float)):
                holes = [holes]
            base, tp = p["base"], p["top"]
            kind = p["kind"]
            for j in range(j0, j1 + 1):
                y = self.miny + (j + 0.5) * CELL
                for i in range(i0, i1 + 1):
                    x = self.minx + (i + 0.5) * CELL
                    if not _inside(x, y, ring):
                        continue
                    if any(_inside(x, y, h) for h in holes):
                        continue
                    k = j * nx + i
                    if kind != "bridge" and tp > top[k]:
                        top[k] = tp
                    g = self.ground(x, y)
                    if base < g + 9.0 and tp > g + self.maxstep:
                        raw[k] = 1
        # map edge walls
        for i in range(nx):
            raw[i] = 1
            raw[(ny - 1) * nx + i] = 1
        for j in range(ny):
            raw[j * nx] = 1
            raw[j * nx + nx - 1] = 1
        self.raw = raw
        self.top = top
        blk = bytearray(raw)
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                k = j * nx + i
                if raw[k]:
                    continue
                if (raw[k - 1] or raw[k + 1] or raw[k - nx] or raw[k + nx]
                        or raw[k - nx - 1] or raw[k - nx + 1] or raw[k + nx - 1] or raw[k + nx + 1]):
                    blk[k] = 1
        self.blk = blk

    # ---------- terrain ----------
    def ground(self, x, y):
        fx = (x - self.tminx) / self.tstep
        fy = (y - self.tminy) / self.tstep
        i = int(math.floor(fx))
        j = int(math.floor(fy))
        i = min(max(i, 0), self.tnx - 2)
        j = min(max(j, 0), self.tny - 2)
        tx = min(max(fx - i, 0.0), 1.0)
        ty = min(max(fy - j, 0.0), 1.0)
        t = self.terr
        n = self.tnx
        h00 = t[j * n + i]
        h10 = t[j * n + i + 1]
        h01 = t[(j + 1) * n + i]
        h11 = t[(j + 1) * n + i + 1]
        return 0.01 * ((h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty)

    # ---------- cells ----------
    def cell(self, x, y):
        i = int((x - self.minx) / CELL)
        j = int((y - self.miny) / CELL)
        i = min(max(i, 0), self.nx - 1)
        j = min(max(j, 0), self.ny - 1)
        return j * self.nx + i

    def center(self, k):
        return (self.minx + (k % self.nx + 0.5) * CELL, self.miny + (k // self.nx + 0.5) * CELL)

    def nearest_free(self, k, rmax=8):
        if not self.blk[k]:
            return k
        nx, ny = self.nx, self.ny
        ci, cj = k % nx, k // nx
        for r in range(1, rmax + 1):
            best = None
            bd = 1e9
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = ci + di, cj + dj
                    if 0 <= i < nx and 0 <= j < ny and not self.blk[j * nx + i]:
                        d = di * di + dj * dj
                        if d < bd:
                            bd, best = d, j * nx + i
            if best is not None:
                return best
        return None

    # ---------- queries ----------
    def walk_clear(self, a, b):
        """Straight walk between two world points on the dilated grid."""
        if not self.ok:
            return True
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        n = int(d / 1.5) + 1
        blk, nx = self.blk, self.nx
        for s in range(1, n + 1):
            f = s / n
            i = int((a[0] + dx * f - self.minx) / CELL)
            j = int((a[1] + dy * f - self.miny) / CELL)
            if i < 0 or j < 0 or i >= self.nx or j >= self.ny or blk[j * nx + i]:
                return False
        return True

    def los(self, a, b):
        """Sight/shot line between 3D points a, b vs buildings (absolute z)."""
        if not self.ok:
            return True
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.hypot(dx, dy)
        n = int(d / 2.0) + 1
        top, nx = self.top, self.nx
        for s in range(1, n):
            f = s / n
            i = int((a[0] + dx * f - self.minx) / CELL)
            j = int((a[1] + dy * f - self.miny) / CELL)
            if 0 <= i < nx and 0 <= j < self.ny:
                if top[j * nx + i] > a[2] + dz * f:
                    return False
        return True

    def search(self, start_xy, goal_xy):
        return Search(self, start_xy, goal_xy)


class Search:
    def __init__(self, nav, s, g):
        self.nav = nav
        self.done = False
        self.path = None
        self.fail = False
        self.goal_xy = g
        sk = nav.nearest_free(nav.cell(*s))
        gk = nav.nearest_free(nav.cell(*g))
        if sk is None or gk is None:
            self.done = self.fail = True
            return
        self.sk, self.gk = sk, gk
        self.gi, self.gj = gk % nav.nx, gk // nav.nx
        self.gs = {sk: 0.0}
        self.came = {}
        self.closed = set()
        self.open = [(0.0, sk)]
        self.expanded = 0

    def h(self, k):
        nx = self.nav.nx
        dx = abs(k % nx - self.gi)
        dy = abs(k // nx - self.gj)
        return (max(dx, dy) + 0.414 * min(dx, dy))

    def step(self, deadline):
        if self.done:
            return
        nav = self.nav
        nx, ny, blk = nav.nx, nav.ny, nav.blk
        gs, came, closed, op = self.gs, self.came, self.closed, self.open
        gk = self.gk
        while op:
            for _ in range(48):
                if not op:
                    break
                f, k = heapq.heappop(op)
                if k in closed:
                    continue
                if k == gk:
                    self._finish()
                    return
                closed.add(k)
                self.expanded += 1
                ci, cj = k % nx, k // nx
                gk0 = gs[k]
                for di, dj, c in DIRS:
                    i, j = ci + di, cj + dj
                    if i < 0 or j < 0 or i >= nx or j >= ny:
                        continue
                    m = j * nx + i
                    if blk[m] or m in closed:
                        continue
                    if di and dj and (blk[cj * nx + i] or blk[j * nx + ci]):
                        continue
                    ng = gk0 + c
                    if ng < gs.get(m, 1e18):
                        gs[m] = ng
                        came[m] = k
                        hh = max(abs(i - self.gi), abs(j - self.gj)) + 0.414 * min(abs(i - self.gi), abs(j - self.gj))
                        heapq.heappush(op, (ng + 1.6 * hh, m))
            if time.perf_counter() > deadline:
                return
        self.done = self.fail = True

    def _finish(self):
        nav = self.nav
        k = self.gk
        cells = [k]
        while k in self.came:
            k = self.came[k]
            cells.append(k)
        cells.reverse()
        pts = [nav.center(c) for c in cells]
        pts[-1] = self.goal_xy if nav.walk_clear(pts[-1], self.goal_xy) else pts[-1]
        self.path = pts
        self.done = True
