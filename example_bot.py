"""Survivor bot: stay in the zone, loot smartly, fight with lead-aimed weapons, dodge."""
import math
import os
import random
import sys
import time

import mechfight as mf
from nav import Nav

TAU = math.pi * 2


def adiff(a, b):
    d = (a - b + math.pi) % TAU - math.pi
    return d


def norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-6 else (0.0, 0.0)


UP_PRIORITY = ["armor", "plating", "laser_damage", "gun_damage", "laser_charge", "nanites",
               "servos", "gun_rate", "gyros", "lightning_efficiency", "mortar_reload", "sensors",
               "gun_mag", "capacitor", "jets", "reactor", "lightning_chain", "gun_velocity",
               "mortar_splash", "lightning_range", "mortar_velocity", "laser_pierce"]
WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}


class Bot:
    name = "aggressor-py"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(hello.get("seed", 0))
        mapdir = hello.get("map", {}).get("dir", "/arena")
        self.nav = Nav(mapdir, self.cfg["mech"])
        self.W = self.cfg["weapons"]
        self.search = None
        self.path = None
        self.path_goal = None
        self.path_t = -99
        self.pi = 0
        self.pick_mem = {}      # id -> (kind,pos)
        self.cube_mem = {}      # id -> (pos, t)
        self.crate_mem = {}
        self.bad = {}           # goal key -> until t
        self.bad_ids = {}
        self.hist = []
        self.unstick_until = -1
        self.unstick_dir = (1, 0)
        self.strafe = 1
        self.strafe_until = 0
        self.last_hit_t = -99
        self.last_hit_bearing = 0.0
        self.look = None
        self.boosting = False
        self.last_move = (0.0, 0.0)
        self.retreating = False
        self.last_yaw = 0.0

    # ------------------------------------------------------------------
    def ground(self, x, y):
        return self.nav.ground(x, y) if self.nav.ok else 0.0

    def act(self, obs):
        t0 = time.perf_counter()
        me = obs["self"]
        t = obs["t"]
        px, py, pz = me["pos"]
        stats = me["stats"]
        scale = 1 + 0.05 * me.get("cubes", 0)
        self.update_memory(obs, me, t)
        act = {"tick": obs["tick"]}

        # upgrades
        if obs.get("offer"):
            ch = [c["id"] for c in obs["offer"]["choices"]]
            pick = 0
            for u in UP_PRIORITY:
                if u in ch:
                    pick = ch.index(u)
                    break
            act["pick"] = pick

        wk = [w["kind"] if w else None for w in me["weapons"]]

        # ---- zone / goal ----
        z = obs["zone"]
        safe_pt, urgent, in_target = self.safe_point(obs, me, t)

        # ---- enemies ----
        eye = (px, py, pz + stats.get("eye_height", 8))
        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        enemies = []
        for m in mechs:
            sc = 1 + 0.05 * m.get("cubes", 0)
            c = (m["pos"][0], m["pos"][1], m["pos"][2] + 4.5 * sc)
            d = math.dist(m["pos"][:2], (px, py))
            los = self.nav.los(eye, c)
            enemies.append((m, d, los, c, sc))
        vis = [e for e in enemies if e[2] or e[1] < 45]
        flee_vec = None
        hpf = me["hp"] / max(1.0, me["max_hp"])
        threats = [e for e in enemies if (e[2] and e[1] < 200) or e[1] < 40]
        avoid = False
        if threats:
            strongest = max(e[0].get("hp", 1000) for e in threats)
            nearest = min(e[1] for e in threats)
            if self.retreating:
                if hpf > 0.6 or strongest < me["hp"] * 0.8:
                    self.retreating = False
            elif hpf < 0.5 and strongest >= me["hp"]:
                self.retreating = True
            avoid = self.retreating and nearest >= 35
        elif hpf > 0.6:
            self.retreating = False

        # ---- loot goal ----
        goal = safe_pt
        goal_kind = "zone"
        loot = None
        if not urgent and not avoid and not (hpf < 0.5 and t - self.last_hit_t < 4):
            loot = self.pick_loot(obs, me, t, safe_pt, wk, enemies)
            if loot:
                goal = loot[0]
                goal_kind = loot[1]
        if not urgent and not avoid:
            if enemies:
                tg = min(enemies, key=lambda e: e[1])
                want_d = 25 if (("lightning" in wk) and ("laser" not in wk)) else (90 if ("laser" in wk or "mortar" in wk) else 60)
                if tg[1] > want_d and not (loot and loot[1] == "pickup" and math.dist(loot[0], (px, py)) < 40):
                    goal = (tg[0]["pos"][0], tg[0]["pos"][1])
                    goal_kind = "hunt"
            elif loot is None and t > 15:
                snd = None
                for sd in obs.get("sounds", []):
                    if sd["kind"] in ("gun", "laser", "mortar", "explosion", "lightning") and (snd is None or sd["age"] < snd["age"]):
                        snd = sd
                if snd is not None:
                    self.hunt_snd = (px + 55 * math.cos(snd["bearing"]), py + 55 * math.sin(snd["bearing"]), t + 4)
                hs = getattr(self, "hunt_snd", None)
                if hs and t < hs[2]:
                    goal = (hs[0], hs[1])
                    goal_kind = "hunt"
        # hold position if already safe and nothing to do
        holding = (goal_kind == "zone" and in_target)

        # ---- target selection ----
        target = None
        if enemies:
            def score(e):
                m, d, los, c, sc = e
                hpf = m.get("hp", 1000) / max(1.0, m.get("max_hp", 1400))
                return d * (0.6 + 0.6 * hpf) + (0 if los else 120)
            target = min(enemies, key=score)
        # secondary targets
        sec = None  # (pos, kind)
        if not target or (not target[2] and target[1] > 60):
            for n in npcs:
                d = math.dist(n["pos"][:2], (px, py))
                c = (n["pos"][0], n["pos"][1], n["pos"][2] + 1.7)
                if d < 55 and self.nav.los(eye, c):
                    sec = (c, "npc", d, n)
                    break
            if sec is None and goal_kind in ("crate", "zone") or (sec is None and not urgent):
                for cr in obs["crates"]:
                    c = (cr["pos"][0], cr["pos"][1], cr["pos"][2] + 1.5)
                    d = math.dist(c[:2], (px, py))
                    if d < 70 and self.nav.los(eye, c):
                        sec = (c, "crate", d, cr)
                        break

        # ---- aiming / firing ----
        fire = [False, False]
        aim = None
        gun = lap = None
        for i, w in enumerate(me["weapons"]):
            pass
        slot = {k: i for i, k in enumerate(wk) if k}
        wobs = {w["kind"]: w for w in me["weapons"] if w}
        hunting = None
        if target is not None and (target[2] or target[1] < 45 or "mortar" in slot):
            hunting = target
        elif sec is not None:
            hunting = None
        mode = None
        if hunting is not None:
            m, d, los, c, sc = hunting
            origin = (px, py, pz + stats.get("muzzle_height", 7))
            vel = m["vel"]
            lw = wobs.get("laser")
            lt = wobs.get("lightning")
            mw = wobs.get("mortar")
            short_until = getattr(self, "laser_short_until", -1)
            if lw and (lw.get("charge", 0) > 0 or (lw["cooldown"] <= 0.25 and t > short_until and los and d < lw["stats"].get("range", 280) - 20)):
                mode = "laser"
            elif lt and d < lt["stats"].get("range", 45) - 4 and los and me["energy"] > 6:
                mode = "lightning"
            elif mw and mw["cooldown"] <= 0.3 and 30 < d < 205 and (not los or d > 110):
                mode = "mortar"
            elif los and d < wobs["gun"]["stats"]["range"] - 10 if "gun" in wobs else False:
                mode = "gun"
            if mode == "laser":
                lead = 0.12
                tp = (c[0] + vel[0] * lead, c[1] + vel[1] * lead, c[2])
                aim = self.dir_to(origin, tp)
            elif mode == "lightning":
                aim = self.dir_to(origin, c)
            elif mode == "gun":
                g = wobs["gun"]["stats"]
                tp = self.lead(origin, c, vel, g["speed"])
                aim = self.dir_to(origin, tp)
            elif mode == "mortar":
                aim = self.mortar_aim((px, py, pz + 9.5 * scale), c, vel, mw["stats"], not los)
            if aim is not None:
                ay, ap = aim
                dyaw = abs(adiff(me["aim"][0], ay))
                dpit = abs(me["aim"][1] - ap)
                tol = max(0.03, min(0.15, math.atan2(2.2 * sc, max(d, 1))))
                aligned = dyaw < tol and dpit < tol * 1.3
                if mode == "gun" and "gun" in slot:
                    g = wobs["gun"]
                    if aligned and g.get("ammo", 1) > 0 and g["cooldown"] <= 0.5:
                        fire[slot["gun"]] = True
                elif mode == "laser":
                    lw = wobs["laser"]
                    ch = lw.get("charge", 0)
                    if ch > 0 or aligned:
                        fire[slot["laser"]] = True
                        if ch > 0.2 and not aligned and ch < 0.7:
                            pass
                elif mode == "lightning":
                    if dyaw < 0.35:
                        fire[slot["lightning"]] = True
                elif mode == "mortar":
                    if dyaw < 0.04 and dpit < 0.05:
                        fire[slot["mortar"]] = True
                # gun as bonus when aimed in direction regardless of mode
                if mode in ("laser", "lightning") and "gun" in wobs and los:
                    g = wobs["gun"]
                    if aligned and g.get("ammo", 1) > 0 and g["cooldown"] <= 0.5 and mode == "lightning":
                        fire[slot["gun"]] = True
        elif sec is not None:
            c, kind, d, o = sec
            origin = (px, py, pz + stats.get("muzzle_height", 7))
            if kind == "npc":
                tp = self.lead(origin, c, o["vel"], wobs["gun"]["stats"]["speed"]) if "gun" in wobs else c
            else:
                tp = c
            aim = self.dir_to(origin, tp)
            if "gun" in wobs:
                dyaw = abs(adiff(me["aim"][0], aim[0]))
                g = wobs["gun"]
                if dyaw < 0.06 and g.get("ammo", 1) > 0 and d < 80:
                    fire[slot["gun"]] = True
            mode = "sec-" + kind

        if avoid and threats and min(e[1] for e in threats) > 110:
            fire = [False, False]
            if wobs.get("laser", {}).get("charge", 0) > 0:
                fire = [slot["laser"] == 0, slot["laser"] == 1]  # already charging: let it fire
        # heal out of sight: regen is 45 HP/s after 5 s without combat
        heal_hold = (self.retreating and not urgent and me["in_zone"] and not threats
                     and t - self.last_hit_t > 0.5)
        if heal_hold:
            holding = True
        # ---- movement ----
        mv = self.navigate(obs, me, goal, t, holding, urgent)
        # combat modulation
        strafe_vec = (0.0, 0.0)
        hpf = me["hp"] / max(1.0, me["max_hp"])
        close_threat = [e for e in enemies if e[1] < 160 and (e[2] or e[1] < 50)]
        if close_threat:
            e = min(close_threat, key=lambda e: e[1])
            ex, ey = e[0]["pos"][0] - px, e[0]["pos"][1] - py
            dirn = norm((ex, ey))
            if t > self.strafe_until:
                self.strafe = self.rng.choice([-1, 1])
                self.strafe_until = t + self.rng.uniform(0.8, 2.0)
            perp = (-dirn[1] * self.strafe, dirn[0] * self.strafe)
            weak = avoid
            if weak:
                flee_vec = (-dirn[0], -dirn[1])
                cov = self.cover_for(obs, me, threats or close_threat, t)
                if cov is not None:
                    cd = norm((cov[0] - px, cov[1] - py))
                    strafe_vec = (cd[0] * 1.3 + perp[0] * 0.2, cd[1] * 1.3 + perp[1] * 0.2)
                else:
                    strafe_vec = (perp[0] * 0.5 + flee_vec[0] * 0.9, perp[1] * 0.5 + flee_vec[1] * 0.9)
            else:
                want = 100 if (wobs.get("laser") or wobs.get("mortar")) else 70
                if "lightning" in wobs and "laser" not in wobs:
                    want = 25
                rad = 0.0
                if e[1] < want * 0.6:
                    rad = -0.7
                elif e[1] > want * 1.2 and not urgent:
                    rad = 0.8
                strafe_vec = (perp[0] * 0.8 + dirn[0] * rad, perp[1] * 0.8 + dirn[1] * rad)
            blend = 0.5 if urgent else 1.0
            mv = (mv[0] * (0.6 if not weak else 0.25) + strafe_vec[0] * blend, mv[1] * (0.6 if not weak else 0.25) + strafe_vec[1] * blend)
        # dodging
        dv = self.dodge(obs, me, enemies)
        mv = (mv[0] + dv[0] * 1.8, mv[1] + dv[1] * 1.8)
        mvn = norm(mv)
        mag = min(1.0, math.hypot(*mv)) if (mv[0] or mv[1]) else 0.0
        if holding and not dv[0] and not dv[1] and not close_threat:
            mag = 0.0
        move = [mvn[0] * (1 if mag > 0.05 else 0), mvn[1] * (1 if mag > 0.05 else 0)]

        # stuck detection
        self.hist.append((t, px, py, mag > 0.05))
        self.hist = [h for h in self.hist if t - h[0] < 1.5]
        if t > self.unstick_until and len(self.hist) >= 12 and all(h[3] for h in self.hist) \
                and math.hypot(px - self.hist[0][1], py - self.hist[0][2]) < 3.0 and me["grounded"] and not any(f for f in fire):
            self.unstick_until = t + 1.0
            a = math.atan2(move[1], move[0]) + self.rng.choice([-1, 1]) * self.rng.uniform(1.0, 2.2)
            self.unstick_dir = (math.cos(a), math.sin(a))
            self.path = None
            self.search = None
            self.bad[self.gkey(goal)] = t + 20
        jump = False
        if t < self.unstick_until:
            move = list(self.unstick_dir)
            jump = (obs["tick"] % 6 == 0)

        # boost logic
        boost = False
        energy = me["energy"]
        travelling = (not holding) and (mag > 0.05)
        want_boost = (not me["in_zone"]) or urgent or (flee_vec is not None and close_threat and close_threat[0][1] < 120) \
            or (goal_kind != "zone" and math.dist(goal, (px, py)) > 70) or (math.dist(goal, (px, py)) > 100)
        if self.boosting:
            if energy < 8:
                self.boosting = False
        else:
            if energy > 35 or (not me["in_zone"] and energy > 12):
                self.boosting = True
        lasercharging = wobs.get("laser", {}).get("charge", 0) > 0 if "laser" in wobs else False
        boost = bool(self.boosting and want_boost and travelling and not lasercharging)
        if mode == "lightning" and energy < 25:
            boost = False
        if not me["grounded"] and t < self.unstick_until:
            boost = True

        # ---- default aim when not engaging ----
        if aim is None:
            aim = self.scan_aim(obs, me, move, t, holding)
        act["aim"] = [aim[0], max(-0.7, min(1.3, aim[1]))]

        # ---- weapon pickup ----
        take = self.try_take(obs, me, wk, t)
        if take:
            act["take"] = take[0]
            if take[1] is not None:
                act["drop"] = take[1]
        # reload
        if "gun" in wobs and not any(fire) and wobs["gun"].get("ammo", 99) < wobs["gun"]["stats"].get("mag", 12) * 0.5 and wobs["gun"].get("reload", 0) == 0:
            act["reload"] = True

        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = boost
        act["jump"] = jump
        act["fire"] = fire

        # events
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                mf.log(f"HIT t={t:.1f} hp={me['hp']:.0f} -{ev['amount']:.0f} {ev['weapon']} by={ev.get('by')} rear={ev.get('rear')} goal={goal_kind} avoid={avoid} heal={heal_hold} n_threat={len(threats)} zone={me['in_zone']}")
            if ev.get("type") == "effect" and ev.get("effect") == "short":
                self.laser_short_until = t + 2.0
            if ev.get("type") == "take_failed":
                self.bad_ids[ev["id"]] = t + 60
        mf.draw.label(f"{goal_kind} {mode or ''} {'AVOID' if avoid else ''}{' HEAL' if heal_hold else ''} hp{me['hp']:.0f}")
        mf.draw.point([goal[0], goal[1], pz], "#40c0ff", 6)
        el = (time.perf_counter() - t0) * 1000
        if el > 35:
            mf.log(f"slow act {el:.1f}ms")
        return act

    # ------------------------------------------------------------------
    def gkey(self, g):
        return (int(g[0] // 12), int(g[1] // 12))

    def cover_for(self, obs, me, threats, t):
        key = tuple(sorted(e[0]["id"] for e in threats))
        c = getattr(self, "_cover", None)
        px, py = me["pos"][0], me["pos"][1]
        if c and c[0] == key and t - c[1] < 0.6 and math.hypot(c[2][0] - px, c[2][1] - py) > 3:
            return c[2]
        res = self.find_cover(obs, px, py, threats)
        self._cover = (key, t, res) if res else None
        return res

    def find_cover(self, obs, px, py, threats):
        nav = self.nav
        if not nav.ok:
            return None
        z = obs["zone"]
        best, bs = None, 1e9
        for r in (10, 18, 28, 40):
            for k in range(16):
                a = k * TAU / 16
                x, y = px + r * math.cos(a), py + r * math.sin(a)
                if nav.blk[nav.cell(x, y)] or not nav.walk_clear((px, py), (x, y)):
                    continue
                if math.hypot(x - z["next_center"][0], y - z["next_center"][1]) > max(z["next_radius"], z["radius"]) - 8:
                    continue
                gz = nav.ground(x, y) + 4.5
                if any(nav.los((e[0]["pos"][0], e[0]["pos"][1], e[0]["pos"][2] + 8), (x, y, gz)) for e in threats):
                    continue
                dmin = min(math.hypot(x - e[0]["pos"][0], y - e[0]["pos"][1]) for e in threats)
                sc = r + 0.6 * max(0.0, 70 - dmin)
                if sc < bs:
                    bs, best = sc, (x, y)
            if best is not None:
                break
        return best

    def dir_to(self, o, p):
        dx, dy, dz = p[0] - o[0], p[1] - o[1], p[2] - o[2]
        return (math.atan2(dy, dx), math.atan2(dz, max(math.hypot(dx, dy), 1e-6)))

    def lead(self, o, c, v, speed):
        tt = 0.0
        p = c
        for _ in range(3):
            p = (c[0] + v[0] * tt, c[1] + v[1] * tt, c[2] + v[2] * tt * 0.0)
            tt = math.dist(o, p) / speed
        return p

    def mortar_aim(self, o, c, v, ms, high):
        sp, g = ms["speed"], self.W["mortar"]["gravity"]
        tt = 1.0
        p = c
        th = 0.7
        for _ in range(3):
            p = (c[0] + v[0] * tt, c[1] + v[1] * tt, c[2])
            x = math.hypot(p[0] - o[0], p[1] - o[1])
            y = p[2] - o[2]
            disc = sp ** 4 - g * (g * x * x + 2 * y * sp * sp)
            if disc < 0 or x < 1:
                disc = max(disc, 0.0)
            s = math.sqrt(disc)
            th = math.atan((sp * sp + (s if high else -s)) / (g * max(x, 1)))
            tt = x / max(sp * math.cos(th), 1e-3)
        yaw = math.atan2(p[1] - o[1], p[0] - o[0])
        return (yaw, th)

    # ------------------------------------------------------------------
    def update_memory(self, obs, me, t):
        px, py = me["pos"][0], me["pos"][1]
        seen = set()
        for p in obs["pickups"]:
            self.pick_mem[p["id"]] = (p["kind"], p["pos"])
            seen.add(p["id"])
        for k in list(self.pick_mem):
            kind, pos = self.pick_mem[k]
            if k not in seen and math.hypot(pos[0] - px, pos[1] - py) < 20:
                del self.pick_mem[k]
        seenc = set()
        for c in obs["cubes"]:
            self.cube_mem[c["id"]] = (c["pos"], t)
            seenc.add(c["id"])
        for k in list(self.cube_mem):
            pos, ts = self.cube_mem[k]
            if t - ts > 20 or (k not in seenc and math.hypot(pos[0] - px, pos[1] - py) < 15):
                del self.cube_mem[k]
        seenr = set()
        for c in obs["crates"]:
            self.crate_mem[c["id"]] = (c["pos"], t)
            seenr.add(c["id"])
        for k in list(self.crate_mem):
            pos, ts = self.crate_mem[k]
            if k not in seenr and math.hypot(pos[0] - px, pos[1] - py) < 30:
                del self.crate_mem[k]
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged" and ev.get("bearing") is not None:
                self.last_hit_t = t
                self.last_hit_bearing = ev["bearing"]
            if ev.get("type") == "took":
                self.pick_mem.pop(ev["id"], None)

    def safe_point(self, obs, me, t):
        z = obs["zone"]
        px, py = me["pos"][0], me["pos"][1]
        cx, cy = z["next_center"]
        r = z["next_radius"]
        # also respect current circle if it is smaller (mid-shrink)
        if z["radius"] < r + math.hypot(z["center"][0] - cx, z["center"][1] - cy):
            pass
        want_r = r * 0.5 if r > 30 else 0.0
        dx, dy = px - cx, py - cy
        d = math.hypot(dx, dy)
        if d <= max(want_r * 1.15, 8):
            pt = (px, py)
            inside = True
        else:
            u = (dx / d, dy / d)
            pt = (cx + u[0] * want_r, cy + u[1] * want_r)
            inside = d <= r * 0.8
        # urgency: time to get there vs time before circle edge reaches us
        dist = math.hypot(pt[0] - px, pt[1] - py)
        T = dist * 1.35 / 11.7
        if t < z["shrink_end"]:
            A = z["shrink_end"] - t - 6
        else:
            A = 30
        cur_d = math.hypot(px - z["center"][0], py - z["center"][1])
        out_now = cur_d > z["radius"] - 6
        urgent = out_now or (t >= z["shrink_start"] - 35 and dist > 5 and T > A * 0.7) or (not me["in_zone"])
        if t < 10 and not out_now:
            urgent = False
        return pt, urgent, inside and not out_now

    def pick_loot(self, obs, me, t, safe, wk, enemies):
        px, py, pz = me["pos"]
        z = obs["zone"]
        A = (z["shrink_end"] - t - 8) if t < z["shrink_end"] else 25
        A = max(A, 8)
        cands = []
        have2 = wk[1] is not None
        best_rank = max([WEAPON_RANK.get(k, 0) for k in wk[1:] if k] or [-1])
        for pid, (kind, pos) in self.pick_mem.items():
            if kind == "gun" or self.bad_ids.get(pid, -1) > t:
                continue
            if kind in wk:
                continue
            rank = WEAPON_RANK[kind]
            if have2 and rank <= best_rank:
                continue
            if pos[2] - self.ground(pos[0], pos[1]) > 3:
                continue
            d = math.hypot(pos[0] - px, pos[1] - py)
            if d > 160:
                continue
            cands.append((d - 25 * rank, (pos[0], pos[1]), "pickup", pid, d))
        for cid, (pos, ts) in self.cube_mem.items():
            if self.bad_ids.get(cid, -1) > t or me.get("cubes", 0) >= 10:
                continue
            d = math.hypot(pos[0] - px, pos[1] - py)
            if d < 70:
                cands.append((d - 10, (pos[0], pos[1]), "cube", cid, d))
        for cid, (pos, ts) in self.crate_mem.items():
            if self.bad_ids.get(cid, -1) > t:
                continue
            if pos[2] - self.ground(pos[0], pos[1]) > 3:
                continue
            d = math.hypot(pos[0] - px, pos[1] - py)
            if d < 90:
                cands.append((d + 10, (pos[0], pos[1]), "crate", cid, d))
        for dr in obs.get("drops", []):
            if self.bad_ids.get(dr["id"], -1) > t:
                continue
            pos = dr["pos"]
            d = math.hypot(pos[0] - px, pos[1] - py)
            others = dr.get("holder") not in (None, me["id"])
            dzone = math.hypot(pos[0] - z["next_center"][0], pos[1] - z["next_center"][1])
            if others and dr.get("progress", 0) > 0:
                continue
            if d / 11.7 + 2 >= dr["lands_in"] - 8 or dr["lands_in"] < 1:
                if d < 260 and dzone < z["next_radius"] + 5:
                    cands.append((d - 120, (pos[0], pos[1]), "drop", dr["id"], d))
        cands.sort(key=lambda c: c[0])
        for score, pos, kind, cid, d in cands[:6]:
            if self.bad.get(self.gkey(pos), -1) > t:
                continue
            extra = (d + math.hypot(pos[0] - safe[0], pos[1] - safe[1])) * 1.4 / 11.7
            if kind == "drop":
                extra *= 0.7
            if extra < A:
                # crates: stop at standoff to shoot them
                if kind == "crate" and d < 22:
                    return ((px, py), "crate", cid)
                return (pos, kind, cid)
        return None

    def try_take(self, obs, me, wk, t):
        px, py, pz = me["pos"]
        best = None
        for p in obs["pickups"]:
            if p["kind"] == "gun" or p["kind"] in wk or self.bad_ids.get(p["id"], -1) > t:
                continue
            d = math.hypot(p["pos"][0] - px, p["pos"][1] - py)
            if d > 3.6 or abs(p["pos"][2] - pz) > 3.5:
                continue
            rank = WEAPON_RANK[p["kind"]]
            if wk[1] is None:
                cand = (rank, p["id"], None)
            else:
                cur = WEAPON_RANK.get(wk[1], 0)
                if rank <= cur:
                    continue
                cand = (rank, p["id"], 1)
            if best is None or cand[0] > best[0]:
                best = cand
        if best:
            return (best[1], best[2])
        return None

    # ------------------------------------------------------------------
    def navigate(self, obs, me, goal, t, holding, urgent):
        px, py = me["pos"][0], me["pos"][1]
        d = math.hypot(goal[0] - px, goal[1] - py)
        if holding or d < 2.5:
            return (0.0, 0.0)
        nav = self.nav
        if not nav.ok:
            return norm((goal[0] - px, goal[1] - py))
        # direct walk if clear and short
        if d < 40 and nav.walk_clear((px, py), goal):
            self.path = None
            self.search = None
            return norm((goal[0] - px, goal[1] - py))
        # (re)plan
        need = self.path is None or self.path_goal is None or math.hypot(self.path_goal[0] - goal[0], self.path_goal[1] - goal[1]) > 10 \
            or (t - self.path_t > 4.0)
        if need and (self.search is None or math.hypot(self.search.goal_xy[0] - goal[0], self.search.goal_xy[1] - goal[1]) > 10):
            self.search = nav.search((px, py), goal)
        elif need and self.search is None:
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
            pts = self.path
            # advance index to closest ahead
            best = self.pi
            bd = 1e9
            for i in range(self.pi, min(len(pts), self.pi + 12)):
                dd = (pts[i][0] - px) ** 2 + (pts[i][1] - py) ** 2
                if dd < bd:
                    bd, best = dd, i
            self.pi = best
            tgt = pts[min(self.pi + 1, len(pts) - 1)]
            for i in range(min(len(pts) - 1, self.pi + 14), self.pi, -1):
                if nav.walk_clear((px, py), pts[i]):
                    tgt = pts[i]
                    break
            if self.pi >= len(pts) - 1 and math.hypot(pts[-1][0] - px, pts[-1][1] - py) < 3:
                return (0.0, 0.0)
            return norm((tgt[0] - px, tgt[1] - py))
        return norm((goal[0] - px, goal[1] - py))

    def dodge(self, obs, me, enemies):
        px, py, pz = me["pos"]
        cx, cy, cz = px, py, pz + 4.5
        out = [0.0, 0.0]
        for pr in obs["projectiles"]:
            if pr["kind"] == "mortar":
                # landing prediction
                p = list(pr["pos"])
                v = list(pr["vel"])
                g = self.W["mortar"]["gravity"]
                tt = 0.0
                while tt < 6:
                    v[2] -= g * 0.1
                    p[0] += v[0] * 0.1
                    p[1] += v[1] * 0.1
                    p[2] += v[2] * 0.1
                    tt += 0.1
                    if p[2] <= self.ground(p[0], p[1]) + 0.5:
                        break
                dx, dy = px - p[0], py - p[1]
                dd = math.hypot(dx, dy)
                if dd < 11 and tt < 2.0:
                    u = norm((dx, dy)) if dd > 0.5 else (1.0, 0.0)
                    out[0] += u[0] * 1.5
                    out[1] += u[1] * 1.5
                continue
            p = pr["pos"]
            v = pr["vel"]
            rx, ry, rz = cx - p[0], cy - p[1], cz - p[2]
            vv = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
            if vv < 1:
                continue
            tc = (rx * v[0] + ry * v[1] + rz * v[2]) / vv
            if not (0 < tc < 0.8):
                continue
            mx, my, mz = p[0] + v[0] * tc - cx, p[1] + v[1] * tc - cy, p[2] + v[2] * tc - cz
            if math.sqrt(mx * mx + my * my + mz * mz) < 4.0 and abs(mz) < 7:
                vn = norm((v[0], v[1]))
                perp = (-vn[1], vn[0])
                side = 1 if (perp[0] * -mx + perp[1] * -my) >= 0 else -1
                if abs(mx * perp[0] + my * perp[1]) < 0.3:
                    side = self.strafe
                out[0] += perp[0] * side
                out[1] += perp[1] * side
        # laser telegraphs
        for m, d, los, c, sc in enemies:
            if m.get("charging") and d < 290:
                bx, by = px - m["pos"][0], py - m["pos"][1]
                ang = abs(adiff(math.atan2(by, bx), m["aim"][0]))
                if ang < max(0.06, 4.0 / max(d, 10)) * 2:
                    ay = m["aim"][0]
                    perp = (-math.sin(ay), math.cos(ay))
                    side = self.strafe
                    out[0] += perp[0] * side * 1.5
                    out[1] += perp[1] * side * 1.5
        n = math.hypot(*out)
        if n > 1:
            out = [out[0] / n, out[1] / n]
        return out

    def scan_aim(self, obs, me, move, t, holding):
        px, py, pz = me["pos"]
        # recent hit: face attacker
        if t - self.last_hit_t < 1.5:
            return (self.last_hit_bearing, 0.0)
        # recent loud sounds: look toward them
        best = None
        for s in obs.get("sounds", []):
            if s["kind"] in ("gun", "mortar", "laser", "laser_charge", "lightning", "explosion", "step", "boost", "jet") and s["dist"] != "far":
                if best is None or s["age"] < best["age"]:
                    best = s
        if best is not None:
            self.look = (best["bearing"], t + 1.2)
        if self.look and t < self.look[1]:
            return (self.look[0], 0.0)
        if holding or (move[0] == 0 and move[1] == 0):
            return ((t * 1.3) % TAU - math.pi, 0.0)
        # face movement direction, with occasional glance
        yaw = math.atan2(move[1], move[0])
        return (yaw, 0.0)


mf.run(Bot)