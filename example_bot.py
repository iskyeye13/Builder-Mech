"""Guerrilla MechFight bot: stalk, ambush, disengage, and reposition."""

import math
import random
import time

import mechfight as mf
from nav import Nav


TAU = math.pi * 2

UP_PRIORITY = [
    "armor", "plating", "laser_damage", "gun_damage", "laser_charge",
    "nanites", "servos", "gun_rate", "gyros", "lightning_efficiency",
    "mortar_reload", "sensors", "gun_mag", "capacitor", "jets",
    "reactor", "lightning_chain", "gun_velocity", "mortar_splash",
    "lightning_range", "mortar_velocity", "laser_pierce"
]

WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}


def norm(v):
    n = math.hypot(v[0], v[1])
    if n < 1e-6:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def adiff(a, b):
    return (a - b + math.pi) % TAU - math.pi


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Bot:
    name = "guerrilla-py"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(hello.get("seed", 0))
        self.nav = Nav(
            hello.get("map", {}).get("dir", "/arena"),
            self.cfg["mech"]
        )
        self.W = self.cfg["weapons"]

        self.search = None
        self.path = None
        self.path_goal = None
        self.path_t = -99
        self.pi = 0

        self.pick_mem = {}
        self.cube_mem = {}
        self.crate_mem = {}
        self.bad = {}
        self.bad_ids = {}

        self.hist = []
        self.unstick_until = -1
        self.unstick_dir = (1, 0)

        self.strafe = 1
        self.strafe_until = 0
        self.last_hit_t = -99
        self.last_hit_bearing = 0.0

        self.retreating = False
        self.boosting = False
        self.look = None

        # Guerrilla behavior state
        self.target_id = None
        self.target_lock_until = 0
        self.attack_until = 0
        self.reposition_until = 0
        self.reposition_goal = None
        self.last_attack_pos = None
        self.last_attack_target = None
        self.last_target_seen = 0
        self.last_flank_side = 1
        self.hunt_snd = None

        self._cover = None

    # ---------------------------------------------------------------
    # Basic helpers

    def ground(self, x, y):
        return self.nav.ground(x, y) if self.nav.ok else 0.0

    def gkey(self, p):
        return (int(p[0] // 12), int(p[1] // 12))

    def dir_to(self, origin, target):
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        dz = target[2] - origin[2]
        return (
            math.atan2(dy, dx),
            math.atan2(dz, max(math.hypot(dx, dy), 1e-6))
        )

    def lead(self, origin, target, velocity, speed):
        if speed <= 0:
            return target

        tt = 0.0
        point = target

        for _ in range(3):
            point = (
                target[0] + velocity[0] * tt,
                target[1] + velocity[1] * tt,
                target[2]
            )
            tt = math.dist(origin, point) / speed

        return point

    def mortar_aim(self, origin, target, velocity, stats, high):
        speed = stats["speed"]
        gravity = self.W["mortar"]["gravity"]

        tt = 1.0
        point = target
        angle = 0.7

        for _ in range(3):
            point = (
                target[0] + velocity[0] * tt,
                target[1] + velocity[1] * tt,
                target[2]
            )

            horizontal = math.hypot(
                point[0] - origin[0],
                point[1] - origin[1]
            )
            vertical = point[2] - origin[2]

            disc = speed ** 4 - gravity * (
                gravity * horizontal * horizontal
                + 2 * vertical * speed * speed
            )

            disc = max(0.0, disc)
            root = math.sqrt(disc)

            angle = math.atan(
                (speed * speed + (root if high else -root))
                / (gravity * max(horizontal, 1))
            )

            tt = horizontal / max(speed * math.cos(angle), 1e-3)

        yaw = math.atan2(
            point[1] - origin[1],
            point[0] - origin[0]
        )

        return yaw, angle

    # ---------------------------------------------------------------
    # Memory

    def update_memory(self, obs, me, t):
        px, py = me["pos"][:2]

        seen = set()
        for p in obs["pickups"]:
            self.pick_mem[p["id"]] = (p["kind"], p["pos"])
            seen.add(p["id"])

        for key in list(self.pick_mem):
            kind, pos = self.pick_mem[key]
            if key not in seen and distance(pos, (px, py)) < 20:
                del self.pick_mem[key]

        seen = set()
        for c in obs["cubes"]:
            self.cube_mem[c["id"]] = (c["pos"], t)
            seen.add(c["id"])

        for key in list(self.cube_mem):
            pos, ts = self.cube_mem[key]
            if t - ts > 20 or (
                key not in seen and distance(pos, (px, py)) < 15
            ):
                del self.cube_mem[key]

        seen = set()
        for c in obs["crates"]:
            self.crate_mem[c["id"]] = (c["pos"], t)
            seen.add(c["id"])

        for key in list(self.crate_mem):
            pos, ts = self.crate_mem[key]
            if key not in seen and distance(pos, (px, py)) < 30:
                del self.crate_mem[key]

        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                self.last_hit_t = t
                if ev.get("bearing") is not None:
                    self.last_hit_bearing = ev["bearing"]

            elif ev.get("type") == "took":
                self.pick_mem.pop(ev["id"], None)

    # ---------------------------------------------------------------
    # Zone handling

    def safe_point(self, obs, me, t):
        z = obs["zone"]
        px, py = me["pos"][:2]

        cx, cy = z["next_center"]
        radius = z["next_radius"]

        desired_radius = radius * 0.42 if radius > 30 else 0.0
        dx, dy = px - cx, py - cy
        d = math.hypot(dx, dy)

        if d <= max(desired_radius * 1.15, 8):
            point = (px, py)
            inside = True
        else:
            u = (dx / max(d, 1e-6), dy / max(d, 1e-6))
            point = (
                cx + u[0] * desired_radius,
                cy + u[1] * desired_radius
            )
            inside = d <= radius * 0.8

        travel_time = distance(point, (px, py)) * 1.35 / 11.7

        if t < z["shrink_end"]:
            available = z["shrink_end"] - t - 6
        else:
            available = 30

        current_distance = distance((px, py), z["center"])
        outside_now = current_distance > z["radius"] - 6

        urgent = (
            outside_now
            or (
                t >= z["shrink_start"] - 35
                and distance(point, (px, py)) > 5
                and travel_time > available * 0.7
            )
            or not me["in_zone"]
        )

        if t < 10 and not outside_now:
            urgent = False

        return point, urgent, inside and not outside_now

    # ---------------------------------------------------------------
    # Cover and ambush positioning

    def find_cover(self, obs, px, py, threats):
        if not self.nav.ok or not threats:
            return None

        z = obs["zone"]
        best = None
        best_score = 1e9

        for radius in (12, 20, 30, 42, 55):
            for k in range(20):
                angle = k * TAU / 20
                x = px + radius * math.cos(angle)
                y = py + radius * math.sin(angle)

                if self.nav.blk[self.nav.cell(x, y)]:
                    continue

                if not self.nav.walk_clear((px, py), (x, y)):
                    continue

                zone_d = distance((x, y), z["next_center"])
                if zone_d > max(z["next_radius"], z["radius"]) - 10:
                    continue

                point_z = self.nav.ground(x, y) + 4.5

                blocked = False
                for e in threats:
                    enemy = e[0]
                    enemy_eye = (
                        enemy["pos"][0],
                        enemy["pos"][1],
                        enemy["pos"][2] + 8
                    )
                    if self.nav.los(enemy_eye, (x, y, point_z)):
                        blocked = True
                        break

                if blocked:
                    continue

                nearest = min(
                    distance((x, y), e[0]["pos"])
                    for e in threats
                )

                score = radius + max(0, 90 - nearest) * 0.7

                if score < best_score:
                    best_score = score
                    best = (x, y)

            if best is not None:
                break

        return best

    def cover_for(self, obs, me, threats, t):
        px, py = me["pos"][:2]
        key = tuple(sorted(e[0]["id"] for e in threats))

        cached = self._cover
        if (
            cached
            and cached[0] == key
            and t - cached[1] < 0.7
            and cached[2] is not None
            and distance(cached[2], (px, py)) > 4
        ):
            return cached[2]

        result = self.find_cover(obs, px, py, threats)
        self._cover = (key, t, result)
        return result

    def flank_point(self, obs, me, target, threats):
        px, py = me["pos"][:2]
        tx, ty = target["pos"][:2]

        toward = norm((tx - px, ty - py))
        side = (-toward[1], toward[0])

        # Alternate flank direction so we do not repeat the same approach.
        self.last_flank_side *= -1

        candidates = []
        for radius in (55, 75, 95, 115):
            for side_amount in (0.65, 1.0, 1.35):
                x = tx - toward[0] * radius + (
                    side[0] * radius * side_amount * self.last_flank_side
                )
                y = ty - toward[1] * radius + (
                    side[1] * radius * side_amount * self.last_flank_side
                )

                if self.nav.ok:
                    if self.nav.blk[self.nav.cell(x, y)]:
                        continue
                    if not self.nav.walk_clear((px, py), (x, y)):
                        continue

                z = obs["zone"]
                if distance((x, y), z["next_center"]) > z["next_radius"] - 8:
                    continue

                # Prefer locations that are not visible to the target.
                point_z = self.ground(x, y) + 4.5
                target_eye = (
                    tx, ty, target["pos"][2] + 8
                )
                visible = self.nav.los(target_eye, (x, y, point_z))

                score = distance((x, y), (px, py))
                if visible:
                    score += 80

                # Avoid returning to the same attack location.
                if self.last_attack_pos:
                    score += max(
                        0,
                        45 - distance((x, y), self.last_attack_pos)
                    ) * 1.5

                candidates.append((score, (x, y)))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        return None

    # ---------------------------------------------------------------
    # Target selection

    def choose_target(self, enemies, me, t):
        if not enemies:
            return None

        px, py = me["pos"][:2]

        # Count nearby enemies around each candidate. Isolated targets
        # are safer to attack than targets surrounded by a group.
        scored = []

        for e in enemies:
            enemy, d, los, center, scale = e
            nearby = sum(
                1 for other in enemies
                if other[0]["id"] != enemy["id"]
                and distance(other[0]["pos"], enemy["pos"]) < 105
            )

            hp_ratio = enemy.get("hp", 1000) / max(
                1.0, enemy.get("max_hp", 1400)
            )

            score = d * 0.65
            score += hp_ratio * 65
            score += nearby * 65

            if not los:
                score += 100

            if d < 35:
                score -= 20

            # Keep a target briefly to prevent constant switching.
            if enemy["id"] == self.target_id and t < self.target_lock_until:
                score -= 45

            # If we just attacked this target, favor another one.
            if enemy["id"] == self.last_attack_target:
                score += 30

            scored.append((score, e))

        scored.sort(key=lambda item: item[0])
        chosen = scored[0][1]

        if chosen[0]["id"] != self.target_id:
            self.target_id = chosen[0]["id"]
            self.target_lock_until = t + 1.4

        return chosen

    # ---------------------------------------------------------------
    # Loot

    def pick_loot(self, obs, me, t, safe, weapons, enemies):
        px, py = me["pos"][:2]
        z = obs["zone"]

        available_time = (
            z["shrink_end"] - t - 8
            if t < z["shrink_end"]
            else 25
        )
        available_time = max(available_time, 8)

        candidates = []
        have_second = weapons[1] is not None

        best_rank = max(
            [WEAPON_RANK.get(k, -1) for k in weapons[1:] if k]
            or [-1]
        )

        for pid, (kind, pos) in self.pick_mem.items():
            if kind == "gun" or kind in weapons:
                continue
            if self.bad_ids.get(pid, -1) > t:
                continue

            rank = WEAPON_RANK.get(kind, -1)
            if rank < 0:
                continue

            if have_second and rank <= best_rank:
                continue

            if pos[2] - self.ground(pos[0], pos[1]) > 3:
                continue

            d = distance(pos, (px, py))
            if d < 145:
                candidates.append((
                    d - 25 * rank, (pos[0], pos[1]), "pickup", pid, d
                ))

        for cid, (pos, ts) in self.cube_mem.items():
            if self.bad_ids.get(cid, -1) > t:
                continue
            if me.get("cubes", 0) >= 10:
                continue

            d = distance(pos, (px, py))
            if d < 65:
                candidates.append((
                    d - 10, (pos[0], pos[1]), "cube", cid, d
                ))

        for cid, (pos, ts) in self.crate_mem.items():
            if self.bad_ids.get(cid, -1) > t:
                continue
            if pos[2] - self.ground(pos[0], pos[1]) > 3:
                continue

            d = distance(pos, (px, py))
            if d < 80:
                candidates.append((
                    d + 10, (pos[0], pos[1]), "crate", cid, d
                ))

        for drop in obs.get("drops", []):
            if self.bad_ids.get(drop["id"], -1) > t:
                continue

            pos = drop["pos"]
            d = distance(pos, (px, py))
            zone_d = distance(pos, z["next_center"])

            if drop.get("holder") not in (None, me["id"]):
                continue

            if d / 11.7 + 2 >= drop["lands_in"] - 8:
                if d < 220 and zone_d < z["next_radius"]:
                    candidates.append((
                        d - 100, (pos[0], pos[1]), "drop", drop["id"], d
                    ))

        candidates.sort(key=lambda c: c[0])

        for score, pos, kind, cid, d in candidates[:6]:
            if self.bad.get(self.gkey(pos), -1) > t:
                continue

            travel = (
                d + distance(pos, safe)
            ) * 1.4 / 11.7

            if kind == "drop":
                travel *= 0.7

            if travel < available_time:
                if kind == "crate" and d < 22:
                    return ((px, py), "crate", cid)
                return (pos, kind, cid)

        return None

    def try_take(self, obs, me, weapons, t):
        px, py, pz = me["pos"]
        best = None

        for p in obs["pickups"]:
            kind = p["kind"]

            if kind == "gun" or kind in weapons:
                continue
            if self.bad_ids.get(p["id"], -1) > t:
                continue

            rank = WEAPON_RANK.get(kind, -1)
            if rank < 0:
                continue

            d = distance(p["pos"], (px, py))
            if d > 3.6 or abs(p["pos"][2] - pz) > 3.5:
                continue

            if weapons[1] is None:
                candidate = (rank, p["id"], None)
            else:
                current = WEAPON_RANK.get(weapons[1], -1)
                if rank <= current:
                    continue
                candidate = (rank, p["id"], 1)

            if best is None or candidate[0] > best[0]:
                best = candidate

        if best:
            return best[1], best[2]

        return None

    # ---------------------------------------------------------------
    # Navigation

    def navigate(self, obs, me, goal, t, holding, urgent):
        px, py = me["pos"][:2]
        d = distance(goal, (px, py))

        if holding or d < 2.5:
            return (0.0, 0.0)

        nav = self.nav
        if not nav.ok:
            return norm((goal[0] - px, goal[1] - py))

        if d < 40 and nav.walk_clear((px, py), goal):
            self.path = None
            self.search = None
            return norm((goal[0] - px, goal[1] - py))

        need = (
            self.path is None
            or self.path_goal is None
            or distance(self.path_goal, goal) > 10
            or t - self.path_t > 4.0
        )

        if need and (
            self.search is None
            or distance(self.search.goal_xy, goal) > 10
        ):
            self.search = nav.search((px, py), goal)

        if self.search is not None:
            self.search.step(time.perf_counter() + 0.012)

            if self.search.done:
                if self.search.path:
                    self.path = self.search.path
                    self.path_goal = goal
                    self.path_t = t
                    self.pi = 0
                else:
                    self.bad[self.gkey(goal)] = t + 15
                    self.path = None
                    self.path_goal = goal
                    self.path_t = t

                self.search = None

        if self.path:
            points = self.path
            best = self.pi
            best_d = 1e9

            for i in range(self.pi, min(len(points), self.pi + 12)):
                dd = (
                    (points[i][0] - px) ** 2
                    + (points[i][1] - py) ** 2
                )
                if dd < best_d:
                    best_d = dd
                    best = i

            self.pi = best
            target = points[min(self.pi + 1, len(points) - 1)]

            for i in range(
                min(len(points) - 1, self.pi + 14),
                self.pi,
                -1
            ):
                if nav.walk_clear((px, py), points[i]):
                    target = points[i]
                    break

            if (
                self.pi >= len(points) - 1
                and distance(points[-1], (px, py)) < 3
            ):
                return (0.0, 0.0)

            return norm((target[0] - px, target[1] - py))

        return norm((goal[0] - px, goal[1] - py))

    # ---------------------------------------------------------------
    # Projectile dodging

    def dodge(self, obs, me, enemies):
        px, py, pz = me["pos"]
        center = (px, py, pz + 4.5)
        out = [0.0, 0.0]

        for pr in obs["projectiles"]:
            if pr["kind"] == "mortar":
                p = list(pr["pos"])
                v = list(pr["vel"])
                gravity = self.W["mortar"]["gravity"]
                tt = 0.0

                while tt < 6:
                    v[2] -= gravity * 0.1
                    p[0] += v[0] * 0.1
                    p[1] += v[1] * 0.1
                    p[2] += v[2] * 0.1
                    tt += 0.1

                    if p[2] <= self.ground(p[0], p[1]) + 0.5:
                        break

                dx, dy = px - p[0], py - p[1]
                dd = math.hypot(dx, dy)

                if dd < 11 and tt < 2:
                    u = norm((dx, dy)) if dd > 0.5 else (1, 0)
                    out[0] += u[0] * 1.5
                    out[1] += u[1] * 1.5

                continue

            p = pr["pos"]
            v = pr["vel"]

            rx = center[0] - p[0]
            ry = center[1] - p[1]
            rz = center[2] - p[2]

            vv = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
            if vv < 1:
                continue

            tc = (rx * v[0] + ry * v[1] + rz * v[2]) / vv
            if not (0 < tc < 0.8):
                continue

            mx = p[0] + v[0] * tc - center[0]
            my = p[1] + v[1] * tc - center[1]
            mz = p[2] + v[2] * tc - center[2]

            if math.sqrt(mx * mx + my * my + mz * mz) < 4:
                vn = norm((v[0], v[1]))
                perp = (-vn[1], vn[0])
                side = 1 if perp[0] * -mx + perp[1] * -my >= 0 else -1

                if abs(mx * perp[0] + my * perp[1]) < 0.3:
                    side = self.strafe

                out[0] += perp[0] * side
                out[1] += perp[1] * side

        for enemy, d, los, center_enemy, scale in enemies:
            if enemy.get("charging") and d < 290:
                bx = px - enemy["pos"][0]
                by = py - enemy["pos"][1]
                ang = abs(adiff(
                    math.atan2(by, bx),
                    enemy["aim"][0]
                ))

                if ang < max(0.06, 4.0 / max(d, 10)) * 2:
                    yaw = enemy["aim"][0]
                    perp = (-math.sin(yaw), math.cos(yaw))
                    out[0] += perp[0] * self.strafe * 1.5
                    out[1] += perp[1] * self.strafe * 1.5

        n = math.hypot(*out)
        if n > 1:
            out = [out[0] / n, out[1] / n]

        return out

    # ---------------------------------------------------------------
    # Scanning

    def scan_aim(self, obs, me, move, t, holding):
        if t - self.last_hit_t < 1.5:
            return self.last_hit_bearing, 0.0

        best = None
        for s in obs.get("sounds", []):
            if (
                s["kind"] in (
                    "gun", "mortar", "laser", "laser_charge",
                    "lightning", "explosion", "step", "boost", "jet"
                )
                and s["dist"] != "far"
            ):
                if best is None or s["age"] < best["age"]:
                    best = s

        if best is not None:
            self.look = (best["bearing"], t + 1.2)

        if self.look and t < self.look[1]:
            return self.look[0], 0.0

        if holding or (move[0] == 0 and move[1] == 0):
            return (t * 1.3) % TAU - math.pi, 0.0

        return math.atan2(move[1], move[0]), 0.0

    # ---------------------------------------------------------------
    # Main decision loop

    def act(self, obs):
        started = time.perf_counter()

        me = obs["self"]
        t = obs["t"]
        px, py, pz = me["pos"]
        stats = me["stats"]

        self.update_memory(obs, me, t)

        action = {"tick": obs["tick"]}

        # Choose upgrades.
        if obs.get("offer"):
            choices = [c["id"] for c in obs["offer"]["choices"]]
            for upgrade in UP_PRIORITY:
                if upgrade in choices:
                    action["pick"] = choices.index(upgrade)
                    break

        weapons = [
            w["kind"] if w else None
            for w in me["weapons"]
        ]
        slots = {
            kind: i for i, kind in enumerate(weapons) if kind
        }
        wobs = {
            w["kind"]: w for w in me["weapons"] if w
        }

        # Zone and enemy information.
        safe_point, urgent, in_target = self.safe_point(obs, me, t)

        eye = (
            px, py, pz + stats.get("eye_height", 8)
        )

        mechs = [
            m for m in obs["mechs"]
            if m["kind"] == "mech"
        ]
        npcs = [
            m for m in obs["mechs"]
            if m["kind"] == "npc"
        ]

        enemies = []
        for enemy in mechs:
            scale = 1 + 0.05 * enemy.get("cubes", 0)
            center = (
                enemy["pos"][0],
                enemy["pos"][1],
                enemy["pos"][2] + 4.5 * scale
            )
            d = distance(enemy["pos"], (px, py))
            los = self.nav.los(eye, center)
            enemies.append((enemy, d, los, center, scale))

        visible = [
            e for e in enemies
            if e[2] or e[1] < 45
        ]

        # Threat assessment.
        hp_ratio = me["hp"] / max(1.0, me["max_hp"])

        threats = [
            e for e in enemies
            if (e[2] and e[1] < 200) or e[1] < 50
        ]

        nearby_count = sum(
            1 for e in enemies
            if e[1] < 150 and (e[2] or e[1] < 50)
        )

        strongest = max(
            [e[0].get("hp", 1000) for e in threats]
            or [0]
        )
        nearest_threat = min(
            [e[1] for e in threats] or [9999]
        )

        if self.retreating:
            if hp_ratio > 0.65 or strongest < me["hp"] * 0.75:
                self.retreating = False
        elif hp_ratio < 0.48 and (
            strongest >= me["hp"] or nearby_count >= 2
        ):
            self.retreating = True

        avoid = self.retreating

        # Pick a target.
        target = self.choose_target(enemies, me, t)

        # If target disappears, stop trying to chase it.
        if target is not None:
            self.last_target_seen = t
        elif t - self.last_target_seen > 2.0:
            self.target_id = None
            self.reposition_goal = None

        # Decide whether to attack or reposition.
        target_visible = (
            target is not None
            and (target[2] or target[1] < 45)
        )

        target_enemy = target[0] if target else None
        target_distance = target[1] if target else 9999

        if avoid:
            self.attack_until = 0
            self.reposition_until = 0
            self.reposition_goal = None

        elif target_visible:
            # Start a short attack window when we have a good opportunity.
            if t > self.attack_until and t > self.reposition_until:
                self.attack_until = t + self.rng.uniform(1.4, 2.1)

            # Once the attack window ends, break contact and change angle.
            if t >= self.attack_until and t >= self.reposition_until:
                self.last_attack_pos = (px, py)
                self.last_attack_target = target_enemy["id"]

                self.reposition_goal = self.flank_point(
                    obs, me, target_enemy, threats
                )
                self.reposition_until = t + self.rng.uniform(2.2, 3.2)
                self.attack_until = 0

        # Base movement goal.
        goal = safe_point
        goal_kind = "zone"
        holding = in_target

        # Retreat takes priority over everything except zone urgency.
        if avoid and threats and nearest_threat > 30:
            cover = self.cover_for(obs, me, threats, t)
            if cover is not None:
                goal = cover
                goal_kind = "cover"
            else:
                enemy = min(threats, key=lambda e: e[1])
                away = norm((
                    px - enemy[0]["pos"][0],
                    py - enemy[0]["pos"][1]
                ))
                goal = (
                    px + away[0] * 65,
                    py + away[1] * 65
                )
                goal_kind = "escape"

            holding = False

        elif urgent:
            goal = safe_point
            goal_kind = "zone"
            holding = False

        elif (
            target is not None
            and self.reposition_goal is not None
            and t < self.reposition_until
        ):
            goal = self.reposition_goal
            goal_kind = "flank"
            holding = False

        elif target_visible and t < self.attack_until:
            # During the attack window, move to weapon range but do not
            # charge directly into the enemy.
            desired = 95

            if "lightning" in wobs and "laser" not in wobs:
                desired = 28
            elif "laser" in wobs:
                desired = 110
            elif "mortar" in wobs:
                desired = 125

            if target_distance > desired + 20:
                direction = norm((
                    target_enemy["pos"][0] - px,
                    target_enemy["pos"][1] - py
                ))
                side = (-direction[1], direction[0])

                # Approach at an angle, not straight down the middle.
                goal = (
                    px + direction[0] * 35 + side[0] * 30,
                    py + direction[1] * 35 + side[1] * 30
                )
                goal_kind = "approach"
                holding = False
            else:
                goal = (px, py)
                goal_kind = "ambush"
                holding = True

        else:
            # No immediate engagement: loot or investigate sounds.
            if not urgent and not avoid and nearby_count == 0:
                loot = self.pick_loot(
                    obs, me, t, safe_point, weapons, enemies
                )
            else:
                loot = None

            if loot:
                goal, goal_kind, loot_id = loot
                holding = False

            elif target is not None and not urgent:
                # Move toward a flank point instead of chasing the target.
                if self.reposition_goal is None or t >= self.reposition_until:
                    self.reposition_goal = self.flank_point(
                        obs, me, target_enemy, threats
                    )
                    self.reposition_until = t + 2.5

                if self.reposition_goal is not None:
                    goal = self.reposition_goal
                    goal_kind = "stalk"
                    holding = False

            elif not urgent and not avoid and t > 15:
                sound = None
                for s in obs.get("sounds", []):
                    if (
                        s["kind"] in (
                            "gun", "laser", "mortar",
                            "explosion", "lightning"
                        )
                        and (sound is None or s["age"] < sound["age"])
                    ):
                        sound = s

                if sound is not None:
                    self.hunt_snd = (
                        px + 65 * math.cos(sound["bearing"]),
                        py + 65 * math.sin(sound["bearing"]),
                        t + 4
                    )

                if self.hunt_snd and t < self.hunt_snd[2]:
                    goal = self.hunt_snd[:2]
                    goal_kind = "investigate"
                    holding = False

        # -----------------------------------------------------------
        # Weapon aiming and firing.

        fire = [False, False]
        aim = None
        mode = None

        hunting = None
        if target is not None and (
            target[2] or target[1] < 45 or "mortar" in slots
        ):
            hunting = target

        # During a reposition, still return fire if an enemy is close.
        if hunting is not None:
            enemy, d, los, center, scale = hunting

            origin = (
                px, py, pz + stats.get("muzzle_height", 7)
            )
            velocity = enemy["vel"]

            laser = wobs.get("laser")
            lightning = wobs.get("lightning")
            mortar = wobs.get("mortar")
            gun = wobs.get("gun")

            if (
                laser
                and (
                    laser.get("charge", 0) > 0
                    or (
                        laser["cooldown"] <= 0.25
                        and los
                        and d < laser["stats"].get("range", 280) - 20
                    )
                )
            ):
                mode = "laser"

            elif (
                lightning
                and d < lightning["stats"].get("range", 45) - 4
                and los
                and me["energy"] > 6
            ):
                mode = "lightning"

            elif (
                mortar
                and mortar["cooldown"] <= 0.3
                and 30 < d < 205
                and (not los or d > 110)
            ):
                mode = "mortar"

            elif (
                gun
                and los
                and d < gun["stats"].get("range", 150) - 10
            ):
                mode = "gun"

            if mode == "laser":
                lead_time = 0.12
                point = (
                    center[0] + velocity[0] * lead_time,
                    center[1] + velocity[1] * lead_time,
                    center[2]
                )
                aim = self.dir_to(origin, point)

            elif mode == "lightning":
                aim = self.dir_to(origin, center)

            elif mode == "gun":
                point = self.lead(
                    origin, center, velocity, gun["stats"]["speed"]
                )
                aim = self.dir_to(origin, point)

            elif mode == "mortar":
                aim = self.mortar_aim(
                    (px, py, pz + 9.5),
                    center,
                    velocity,
                    mortar["stats"],
                    not los
                )

            if aim is not None:
                yaw, pitch = aim
                dyaw = abs(adiff(me["aim"][0], yaw))
                dpitch = abs(me["aim"][1] - pitch)

                tolerance = max(
                    0.03,
                    min(0.15, math.atan2(2.2 * scale, max(d, 1)))
                )

                aligned = (
                    dyaw < tolerance
                    and dpitch < tolerance * 1.3
                )

                # Fire during the attack window, or when defending
                # ourselves from a nearby enemy.
                can_attack = (
                    t < self.attack_until
                    or d < 55
                    or avoid
                )

                if can_attack:
                    if mode == "gun" and aligned:
                        if gun.get("ammo", 1) > 0 and gun["cooldown"] <= 0.5:
                            fire[slots["gun"]] = True

                    elif mode == "laser":
                        charge = laser.get("charge", 0)
                        if charge > 0 or aligned:
                            fire[slots["laser"]] = True

                    elif mode == "lightning":
                        if dyaw < 0.35:
                            fire[slots["lightning"]] = True

                    elif mode == "mortar":
                        if dyaw < 0.04 and dpitch < 0.05:
                            fire[slots["mortar"]] = True

                    # Use the gun as a bonus weapon when lightning is
                    # selected and the target is lined up.
                    if (
                        mode == "lightning"
                        and gun
                        and los
                        and aligned
                        and gun.get("ammo", 1) > 0
                        and gun["cooldown"] <= 0.5
                    ):
                        fire[slots["gun"]] = True

        # -----------------------------------------------------------
        # Secondary targets: nearby NPCs and crates.

        sec = None

        if target is None or (
            not target[2] and target[1] > 60
        ):
            for npc in npcs:
                d = distance(npc["pos"], (px, py))
                center = (
                    npc["pos"][0],
                    npc["pos"][1],
                    npc["pos"][2] + 1.7
                )
                if d < 55 and self.nav.los(eye, center):
                    sec = (center, "npc", d, npc)
                    break

        if sec is None and not urgent and not avoid:
            for crate in obs["crates"]:
                center = (
                    crate["pos"][0],
                    crate["pos"][1],
                    crate["pos"][2] + 1.5
                )
                d = distance(center, (px, py))

                if d < 70 and self.nav.los(eye, center):
                    sec = (center, "crate", d, crate)
                    break

        if sec is not None and hunting is None:
            center, kind, d, obj = sec
            origin = (
                px, py, pz + stats.get("muzzle_height", 7)
            )

            if kind == "npc" and "gun" in wobs:
                point = self.lead(
                    origin,
                    center,
                    obj["vel"],
                    wobs["gun"]["stats"]["speed"]
                )
            else:
                point = center

            aim = self.dir_to(origin, point)

            if "gun" in wobs:
                dyaw = abs(adiff(me["aim"][0], aim[0]))
                gun = wobs["gun"]

                if (
                    dyaw < 0.06
                    and gun.get("ammo", 1) > 0
                    and d < 80
                    and not avoid
                ):
                    fire[slots["gun"]] = True

            mode = "secondary-" + kind

        # -----------------------------------------------------------
        # Movement and combat movement.

        movement = self.navigate(
            obs, me, goal, t, holding, urgent
        )

        close_threats = [
            e for e in enemies
            if e[1] < 160 and (e[2] or e[1] < 50)
        ]

        # Keep a consistent strafe direction for longer periods.
        if t > self.strafe_until:
            self.strafe = self.rng.choice([-1, 1])
            self.strafe_until = t + self.rng.uniform(2.0, 3.2)

        if close_threats:
            enemy = min(close_threats, key=lambda e: e[1])
            direction = norm((
                enemy[0]["pos"][0] - px,
                enemy[0]["pos"][1] - py
            ))
            perpendicular = (
                -direction[1] * self.strafe,
                direction[0] * self.strafe
            )

            if avoid:
                cover = self.cover_for(
                    obs, me, threats or close_threats, t
                )

                if cover is not None:
                    cover_dir = norm((
                        cover[0] - px,
                        cover[1] - py
                    ))
                    movement = (
                        movement[0] * 0.2 + cover_dir[0] * 1.4,
                        movement[1] * 0.2 + cover_dir[1] * 1.4
                    )
                else:
                    away = (-direction[0], -direction[1])
                    movement = (
                        movement[0] * 0.2
                        + away[0] * 1.0
                        + perpendicular[0] * 0.3,
                        movement[1] * 0.2
                        + away[1] * 1.0
                        + perpendicular[1] * 0.3
                    )

            elif target_visible and t < self.attack_until:
                # Circle while attacking instead of standing still.
                desired = 100
                if "lightning" in wobs and "laser" not in wobs:
                    desired = 28

                radial = 0.0
                if enemy[1] < desired * 0.65:
                    radial = -0.65
                elif enemy[1] > desired * 1.2:
                    radial = 0.45

                movement = (
                    movement[0] * 0.45
                    + perpendicular[0] * 0.8
                    + direction[0] * radial,
                    movement[1] * 0.45
                    + perpendicular[1] * 0.8
                    + direction[1] * radial
                )

        # Dodge incoming projectiles.
        dodge = self.dodge(obs, me, enemies)
        movement = (
            movement[0] + dodge[0] * 1.8,
            movement[1] + dodge[1] * 1.8
        )

        move_dir = norm(movement)
        move_mag = min(1.0, math.hypot(*movement))

        if (
            holding
            and dodge[0] == 0
            and dodge[1] == 0
            and not close_threats
        ):
            move_mag = 0.0

        move = [
            move_dir[0] if move_mag > 0.05 else 0.0,
            move_dir[1] if move_mag > 0.05 else 0.0
        ]

        # -----------------------------------------------------------
        # Stuck detection.

        self.hist.append((t, px, py, move_mag > 0.05))
        self.hist = [
            h for h in self.hist
            if t - h[0] < 1.5
        ]

        if (
            t > self.unstick_until
            and len(self.hist) >= 12
            and all(h[3] for h in self.hist)
            and distance((px, py), self.hist[0][1:3]) < 3
            and me["grounded"]
            and not any(fire)
        ):
            self.unstick_until = t + 1.0

            angle = (
                math.atan2(move[1], move[0])
                + self.rng.choice([-1, 1])
                * self.rng.uniform(1.0, 2.2)
            )

            self.unstick_dir = (
                math.cos(angle),
                math.sin(angle)
            )

            self.path = None
            self.search = None
            self.bad[self.gkey(goal)] = t + 20

        jump = False
        if t < self.unstick_until:
            move = list(self.unstick_dir)
            jump = obs["tick"] % 6 == 0

        # -----------------------------------------------------------
        # Boost management.

        energy = me["energy"]
        travelling = move_mag > 0.05

        want_boost = (
            not me["in_zone"]
            or urgent
            or (
                avoid
                and nearest_threat < 120
            )
            or (
                distance(goal, (px, py)) > 100
                and goal_kind not in ("ambush",)
            )
        )

        if self.boosting:
            if energy < 8:
                self.boosting = False
        elif energy > 35 or (
            not me["in_zone"] and energy > 12
        ):
            self.boosting = True

        laser_charging = (
            "laser" in wobs
            and wobs["laser"].get("charge", 0) > 0
        )

        boost = bool(
            self.boosting
            and want_boost
            and travelling
            and not laser_charging
        )

        if mode == "lightning" and energy < 25:
            boost = False

        if not me["grounded"] and t < self.unstick_until:
            boost = True

        # -----------------------------------------------------------
        # Default aim.

        if aim is None:
            aim = self.scan_aim(
                obs, me, move, t, holding
            )

        action["aim"] = [
            aim[0],
            max(-0.7, min(1.3, aim[1]))
        ]

        # Pick up weapons only when it is reasonably safe.
        if not avoid and not urgent and not close_threats:
            take = self.try_take(obs, me, weapons, t)
            if take:
                action["take"] = take[0]
                if take[1] is not None:
                    action["drop"] = take[1]

        # Reload.
        if (
            "gun" in wobs
            and not any(fire)
            and wobs["gun"].get("ammo", 99)
            < wobs["gun"]["stats"].get("mag", 12) * 0.5
            and wobs["gun"].get("reload", 0) == 0
        ):
            action["reload"] = True

        action["move"] = [
            round(move[0], 3),
            round(move[1], 3)
        ]
        action["boost"] = boost
        action["jump"] = jump
        action["fire"] = fire

        # -----------------------------------------------------------
        # Events and debugging.

        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                mf.log(
                    f"HIT t={t:.1f} hp={me['hp']:.0f} "
                    f"goal={goal_kind} avoid={avoid} "
                    f"threats={len(threats)}"
                )

            elif (
                ev.get("type") == "effect"
                and ev.get("effect") == "short"
            ):
                self.laser_short_until = t + 2.0

            elif ev.get("type") == "take_failed":
                self.bad_ids[ev["id"]] = t + 60

        mf.draw.label(
            f"{goal_kind} {mode or ''} "
            f"{'RETREAT' if avoid else ''} "
            f"hp{me['hp']:.0f}"
        )
        mf.draw.point(
            [goal[0], goal[1], pz],
            "#40c0ff",
            6
        )

        elapsed = (time.perf_counter() - started) * 1000
        if elapsed > 35:
            mf.log(f"slow act {elapsed:.1f}ms")

        return action


mf.run(Bot)