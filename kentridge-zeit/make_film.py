#!/usr/bin/env python3
"""
"Taktung" – Stop-Motion-Film nach William Kentridge (Zeit als gesellschaftliches Phänomen)

Erzeugt aus dem ersten Bild (erstes_bild.jpg) eine Folge von 210 Einzelbildern
(10 Bilder pro Sekunde = 21 s), in der die Zeichnung fortlaufend ausradiert
und überzeichnet wird, sowie eine Tonspur. Frühere Zustände bleiben als
geisterhafte Spuren stehen.

Aufruf:
    python3 make_film.py                 # alle Bilder + Ton + film.mp4
    python3 make_film.py preview 0 60    # einzelne Bilder nach out/preview/
"""
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
W, H = 1600, 1200
FPS = 10
N_FRAMES = 210
C = (940.0, 592.0)          # Mittelpunkt der Uhr = Nahtstelle der Umarmung
RMAX, NA = 1130, 2880       # Polar-Raster (Radius, Winkel)

# ---------------------------------------------------------------- Zeitplan --
MARK_START = [8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]    # 12 Striche
PIVOT_F = 33
HAND_GROW = [35, 36, 37, 38]                                     # Zeiger entsteht
TICK0 = 44
INTERVALS = [6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1]                       # Beschleunigung
TICKS = [TICK0 + int(sum(INTERVALS[:k])) for k in range(30)]     # Bild je 6°-Schritt
STOP_F = TICKS[-1]                                               # Zeiger steht auf 6
ARM_F0, ARM_F1 = STOP_F + 5, STOP_F + 16                         # letzte Berührung
HAND_ERASE_F = STOP_F + 23
PIVOT_ERASE_F = HAND_ERASE_F + 2
_me = PIVOT_ERASE_F + 1
MARK_ERASE = {}                                                  # gegen den Uhrzeigersinn
for i, k in enumerate([6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7]):
    MARK_ERASE[k] = _me + (2 * i if i < 6 else 12 + (i - 6))
GOLD_F0 = max(MARK_ERASE.values()) + 5
GOLD_F1 = GOLD_F0 + 31

rng = np.random.default_rng(11)


# ------------------------------------------------------------ Hilfsmittel --
def blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s)


def snoise(shape, sigma, seed):
    r = np.random.default_rng(seed)
    n = r.standard_normal(shape).astype(np.float32)
    if sigma > 0:
        n = blur(n, sigma)
    return (n - n.mean()) / (n.std() + 1e-9)


def to_uniform(n):
    flat = n.ravel()
    order = np.argsort(flat, kind="stable")
    u = np.empty_like(flat)
    u[order] = np.linspace(0, 1, flat.size, dtype=np.float32)
    return u.reshape(n.shape)


def noise1d(n, corr, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal(n + 8 * int(corr) + 8).astype(np.float32)
    k = np.exp(-0.5 * (np.arange(-3 * corr, 3 * corr + 1) / corr) ** 2)
    y = np.convolve(x, k / k.sum(), mode="same")[4 * int(corr) + 4:][:n]
    return (y - y.mean()) / (y.std() + 1e-9)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def polar_fwd(img):
    return cv2.warpPolar(img, (RMAX, NA), C, RMAX, cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR)


def polar_inv(pol):
    return cv2.warpPolar(pol, (W, H), C, RMAX,
                         cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)


def clock_pt(theta_deg, r):
    t = math.radians(theta_deg)
    return (C[0] + r * math.sin(t), C[1] - r * math.cos(t))


# --------------------------------------------------- Vorlage analysieren --
def load_base():
    src = cv2.imread(os.path.join(HERE, "erstes_bild.jpg"))[:, :, ::-1].astype(np.float32) / 255.
    h, w = src.shape[:2]
    th = int(round(w * 3 / 4))
    top = (h - th) // 2
    return cv2.resize(src[top:top + th], (W, H), interpolation=cv2.INTER_AREA)


def estimate_paper(base):
    """Sauberes Papier (Beleuchtung) unter der Zeichnung rekonstruieren."""
    lum = base.mean(2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    bg0 = blur(cv2.dilate(blur(lum, 2), k), 30)
    draw = (blur(lum, 1.5) < bg0 * 0.955).astype(np.uint8)
    draw = cv2.dilate(draw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    # nur Papier, das mit dem Blattrand verbunden ist (keine Inseln in der Figur)
    n, lab = cv2.connectedComponents((1 - draw).astype(np.uint8))
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    paper = np.isin(lab, list(border - {0}) if draw[0, 0] else list(border)).astype(np.float32)
    paper *= (1 - draw)
    wsm = blur(paper, 6)
    sm = blur(base * paper[..., None], 6) / np.maximum(wsm[..., None], 1e-4)
    wt = (wsm > 0.6).astype(np.float32)
    imgs, wts = [sm * wt[..., None]], [wt]
    for _ in range(8):
        imgs.append(cv2.pyrDown(imgs[-1]))
        wts.append(cv2.pyrDown(wts[-1]))
    cur = imgs[-1] / np.maximum(wts[-1][..., None], 1e-6)
    for i in range(7, -1, -1):
        up = cv2.pyrUp(cur, dstsize=(imgs[i].shape[1], imgs[i].shape[0]))
        wg = np.clip(wts[i], 0, 1)[..., None]
        cur = (imgs[i] / np.maximum(wts[i][..., None], 1e-6)) * wg + up * (1 - wg)
    P = blur(cur, 10) * (1 - wt[..., None]) + cur * wt[..., None]
    return blur(P, 3), paper


# ------------------------------------------------------------- Kohlestrich --
class Mark:
    """Ein gezeichnetes Element: Dichte (voll) und Geisterspur (radiert)."""

    def __init__(self, x0, y0, p, tooth, grain=0.5, ghost=(1.3, 4.0), pencil=False):
        self.x0, self.y0 = x0, y0
        h, w = p.shape
        t = tooth[y0:y0 + h, x0:x0 + w]
        pc = np.clip(p, 0, 1)
        if pencil:                                   # feine, durchgehende Linie / weiche Wischspur
            self.d = blur((pc * (0.72 + 0.28 * t)).astype(np.float32), 0.6)
        else:
            cov = np.clip((t - (1 - (1 + grain) * pc)) / grain, 0, 1) * sstep(0.03, 0.2, pc)
            self.d = blur((cov * (0.4 + 0.6 * pc)).astype(np.float32), 0.75)
        self.g = (0.7 * blur(self.d, ghost[0]) + 0.3 * blur(self.d, ghost[1])).astype(np.float32)

    def add(self, layer, wgt, ghost=False):
        a = self.g if ghost else self.d
        h, w = a.shape
        layer[self.y0:self.y0 + h, self.x0:self.x0 + w] += wgt * a


def render_stroke(pts, width, pressure, seed, tooth, wobble=1.0, taper=(0.06, 0.12),
                  press_var=0.22, grain=0.5, upto=1.0, pencil=False, narrow=0.0, fade=0.5):
    """Polylinie als Kohle-/Kreidestrich mit Borsten-Struktur. Liefert Mark."""
    r = np.random.default_rng(seed)
    pts = np.asarray(pts, np.float32)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = float(cum[-1])
    n = max(int(L), 2)
    s = np.linspace(0, L, n)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    tx, ty = np.gradient(x), np.gradient(y)
    nrm = np.hypot(tx, ty) + 1e-6
    nx, ny = -ty / nrm, tx / nrm
    wob = noise1d(n, 35, seed + 1) * wobble + noise1d(n, 7, seed + 2) * wobble * 0.25
    x, y = x + nx * wob, y + ny * wob
    u = s / L
    env = sstep(0, taper[0], u) * 0.45 + 0.55
    env *= 1 - fade * sstep(1 - taper[1], 1.0, u)
    env *= 1 + 0.16 * noise1d(n, 30, seed + 3)
    keep = u <= upto + 1e-6
    nb = max(2, int(round(width / 1.5)))
    offs = np.linspace(-width / 2, width / 2, nb) * 0.82 + r.normal(0, 0.18, nb)
    wj = r.uniform(0.55, 1.0, nb)
    wj[r.random(nb) < 0.15] *= 0.45
    m = 14
    x0, y0 = int(max(np.floor(x.min()) - m, 0)), int(max(np.floor(y.min()) - m, 0))
    x1, y1 = int(min(np.ceil(x.max()) + m, W)), int(min(np.ceil(y.max()) + m, H))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return Mark(0, 0, np.zeros((2, 2), np.float32), tooth)
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    for j in range(nb):
        mod = np.clip(1 + press_var * noise1d(n, 22, seed + 10 + j), 0.2, 1.6)
        v = (wj[j] * mod * env)[keep]
        wf = 1 - narrow * u ** 1.4                                # Strich läuft spitz aus
        qx, qy = (x + nx * offs[j] * wf)[keep] - x0, (y + ny * offs[j] * wf)[keep] - y0
        ix, iy = np.floor(qx).astype(int), np.floor(qy).astype(int)
        fx, fy = qx - ix, qy - iy
        for dx, dy, wq in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < acc.shape[1]) & (jy >= 0) & (jy < acc.shape[0])
            np.add.at(acc, (jy[ok], jx[ok]), (v * wq)[ok])
    d = width * 0.82 / max(nb - 1, 1)
    acc = blur(acc, max(0.65, d * 0.62)) * d
    return Mark(x0, y0, pressure * acc, tooth, grain=grain, pencil=pencil)


# ----------------------------------------------------------------- Aufbau --
class Film:
    def __init__(self):
        self.base = load_base()
        self.P, self.paper = estimate_paper(self.base)
        self.OD = -np.log(np.clip(self.base, 1e-3, 1) / np.clip(self.P, 1e-3, 1))
        ODp = np.clip(self.OD, 0, None)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx, dy = xx - C[0], yy - C[1]
        self.r = np.hypot(dx, dy)
        ths = np.degrees(np.arctan2(dx, -dy))                    # -180..180, 0 = oben
        self.tooth = to_uniform(0.7 * snoise((H, W), 1.0, 1) + 0.5 * snoise((H, W), 3.0, 2))

        # Papierkorn für radierte Stellen
        res = self.OD[self.paper > 0.5]
        g = 0.8 * snoise((H, W), 0.7, 3) + 0.6 * snoise((H, W), 1.8, 4)
        g = g / g.std() * float(res.std()) * 0.9
        self.grain = g[..., None] * np.array([1, 1, 1], np.float32)

        # Gold der Vorlage schützen (bleibt immer stehen)
        chroma = self.OD[..., 2] - self.OD[..., 0]
        zone = np.zeros((H, W), np.uint8)
        cv2.fillPoly(zone, [np.array([[905, 215], [985, 215], [985, 360], [1075, 440], [1075, 520],
                                      [1000, 575], [900, 580], [855, 520], [870, 380], [915, 330]])], 1)
        cv2.fillPoly(zone, [np.array([[925, 668], [1168, 668], [1168, 730], [925, 730]])], 1)
        gm = np.clip((chroma - 0.07) / 0.09, 0, 1) * zone
        seam = np.zeros((H, W), np.float32)
        cv2.fillPoly(seam, [np.array([[925, 175], [961, 175], [964, 275], [925, 275]])], 1.0)
        self.goldmask = np.clip(blur(cv2.dilate(gm, np.ones((5, 5), np.uint8)), 2.0) * 1.4
                                + blur(seam, 4.0), 0, 1)

        # Geisterbild der Vorlage: Rest + in Wischrichtung verschleppte Kohle
        pol = polar_fwd(ODp)
        pol = np.roll(pol, NA // 4, axis=0)                      # Zeile 0 = 12 Uhr
        a = 1 - math.exp(-(360 / NA) / 7.0)
        sm = np.zeros_like(pol)
        acc = np.zeros_like(pol[0])
        for i in range(NA):
            acc = acc + a * (pol[i] - acc)
            sm[i] = acc
        st = snoise((NA, RMAX), 0, 5)
        st = cv2.GaussianBlur(st, (0, 0), sigmaX=2.6, sigmaY=55)
        st = st / st.std()
        st2 = cv2.GaussianBlur(snoise((NA, RMAX), 0, 15), (0, 0), sigmaX=9, sigmaY=90)
        streak = np.clip(1 + 0.11 * st + 0.16 * st2 / st2.std(), 0.4, 1.8).astype(np.float32)
        mx = (xx + 7 * snoise((H, W), 45, 20) + 2.5 * snoise((H, W), 12, 21)).astype(np.float32)
        my = (yy + 7 * snoise((H, W), 45, 22) + 2.5 * snoise((H, W), 12, 23)).astype(np.float32)
        self.streak = cv2.remap(polar_inv(np.roll(streak, -NA // 4, axis=0)), mx, my, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        self.streak *= 1 + 0.10 * snoise((H, W), 9, 24)
        smear = cv2.remap(polar_inv(np.roll(sm, -NA // 4, axis=0)), mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        self.ghostOD = (0.27 * blur(ODp, 1.7) ** 0.78 + 0.115 * smear) * self.streak[..., None]
        near = blur(np.maximum(ODp.mean(2), smear.mean(2)), 6)
        self.near = np.clip(near / 0.02, 0, 1)

        # Wann erreicht der Zeiger welchen Punkt? (rechte Blatthälfte)
        jit = 1.7 * snoise((H, W), 26, 6) + 0.8 * snoise((H, W), 7, 7)
        the = np.clip(ths + jit * np.clip(self.r / 250, 0.35, 1), 0.01, 180)
        k = np.clip(np.ceil(the / 6).astype(int), 1, 30)
        self.tpass = np.array(TICKS, np.float32)[k - 1]
        self.wR = sstep(-4, 5, dx + 3.5 * snoise((H, W), 14, 8))

        # Arm der alten Figur links der Mittellinie (die letzte Berührung)
        arm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(arm, [np.array([[948, 556], [900, 576], [868, 611], [848, 655], [831, 700],
                                     [811, 750], [789, 800], [751, 850], [716, 900], [699, 950],
                                     [696, 1000], [699, 1062], [948, 1078]])], 1)
        self.arm = blur(arm.astype(np.float32), 3.5)
        s = np.clip((yy - 560) / 480, 0, 1) + 0.05 * snoise((H, W), 18, 9)
        self.tarm = ARM_F0 + s * (ARM_F1 - ARM_F0 - 4)

        self.build_marks()

    # ...................................................... gezeichnete Teile
    def build_marks(self):
        T = self.tooth
        # Zifferblatt: zwölf Striche
        self.marks = []
        for k in range(12):
            quarter = k % 3 == 0
            th = k * 30 + rng.normal(0, 0.7)
            r1 = (506 if quarter else 520) + rng.normal(0, 3)
            r2 = (562 if quarter else 556) + rng.normal(0, 3)
            pts = [clock_pt(th, r1), clock_pt(th + rng.normal(0, 0.25), r2)]
            wd = 10.5 if quarter else 7.5
            light = render_stroke(pts, wd * 0.8, 0.5, 100 + k, T, wobble=0.7)
            full = render_stroke(pts, wd, 0.9, 100 + k, T, wobble=0.7, taper=(0.1, 0.25))
            self.marks.append((light, full, pts))
        # Achse
        ang = np.linspace(0, 5 * math.pi, 80)
        rad = np.linspace(0.5, 6.5, 80)
        piv = np.stack([C[0] + rad * np.cos(ang), C[1] + rad * np.sin(ang)], 1)
        self.pivot = (render_stroke(piv, 3.2, 0.55, 150, T, wobble=0.2),
                      render_stroke(piv, 4.6, 1.0, 150, T, wobble=0.2))
        # Zeiger in 31 Stellungen (Rötel), Stellung 0 wächst in vier Schritten
        self.hands = []
        for k in range(31):
            th = k * 6 + rng.normal(0, 0.25)
            pts = [clock_pt(th, 15), clock_pt(th + rng.normal(0, 0.15), 250), clock_pt(th, 497 + rng.normal(0, 4))]
            self.hands.append(render_stroke(pts, 6.4, 0.95, 200 + k, T, wobble=0.9,
                                            taper=(0.02, 0.05), grain=0.6))
        p0 = [clock_pt(0, 15), clock_pt(0, 250), clock_pt(0, 497)]
        self.hand_grow = [render_stroke(p0, 6.4, 0.95, 200, T, wobble=0.9, taper=(0.02, 0.05),
                                        grain=0.6, upto=q) for q in (0.22, 0.48, 0.76)]
        self.build_wind()
        self.build_gold()

    def build_wind(self):
        """Haarsträhnen, die der 'Zeitwind' nach links aus dem Bild zieht."""
        T = self.tooth
        starts = [(470, 205), (345, 392), (500, 305), (300, 305), (560, 180), (405, 338), (245, 418),
                  (520, 445), (365, 272), (185, 398), (430, 420), (330, 348), (545, 218), (280, 440),
                  (395, 252), (160, 428), (480, 372)]
        f0 = [3, 18, 32, 44, 54, 62, 69, 75, 81, 86, 91, 96, 100, 104, 108, 112, 115]
        self.wind = []
        r = np.random.default_rng(77)
        for i, ((sx, sy), fs) in enumerate(zip(starts, f0)):
            speed = float(np.clip((fs - 3) / 110, 0, 1))         # 0 ruhig … 1 gehetzt
            L = r.uniform(240, 330) + 300 * speed
            phi0 = r.uniform(0.04, 0.2) - 0.08 * speed
            A, lam, ph = r.uniform(0.06, 0.15), r.uniform(300, 520), r.uniform(0, 6.28)
            s_ = np.arange(0, L, 4.0)
            u = s_ / L
            phi = phi0 + A * np.sin(2 * math.pi * s_ / lam + ph) - 0.32 * u ** 3
            px = sx - np.cumsum(np.cos(phi)) * 4.0
            py = sy + np.cumsum(np.sin(phi)) * 4.0
            nx, ny = np.sin(phi), np.cos(phi)                    # Normale zur Strähne
            wb = r.uniform(10, 22)
            body = r.random() < 0.65
            grow = int(round(7 - 4 * speed))
            life = int(round(15 - 7 * speed))
            subs = []
            if body:
                subs.append((np.stack([px, py], 1), dict(width=wb, pressure=r.uniform(0.3, 0.42),
                             wobble=2.0, taper=(0.15, 0.4), fade=0.9, narrow=0.9, press_var=0.3)))
            for side in ([-1, 1] if body else [0, 0.6][:1 + (r.random() < 0.5)]):
                off = side * wb / 2 * (1 - 0.9 * u ** 1.4) * r.uniform(0.8, 1.15)
                subs.append((np.stack([px + nx * off, py + ny * off], 1),
                             dict(width=r.uniform(2.4, 3.3), pressure=r.uniform(0.5, 0.8), wobble=2.2,
                                  taper=(0.08, 0.3), fade=0.9, press_var=0.35)))
            stages = [[render_stroke(p_, seed=400 + 10 * i + n_, tooth=T, pencil=True,
                                     upto=(j + 1) / grow, **kw) for n_, (p_, kw) in enumerate(subs)]
                      for j in range(grow)]
            end = fs + grow + life
            if end > STOP_F:
                end = STOP_F + 2 + (i % 5) * 2
            self.wind.append((fs, grow, end, stages))

    def build_gold(self):
        """Goldene Umrisslinie um die Leerstelle (wird am Schluss gezeichnet)."""
        ODp = np.clip(self.OD, 0, None).mean(2)
        pol = np.roll(polar_fwd((blur(ODp, 5) > 0.13).astype(np.float32)), NA // 4, axis=0)
        rr = np.arange(RMAX, dtype=np.float32)[None, :]
        rmax = ((pol > 0.5) * rr).max(1)                          # Silhouette je Winkel
        thetas = np.arange(4.0, 171.0, 1.0)
        idx = (thetas / 360 * NA).astype(int)
        prof = np.array([np.median(rmax[i - 14:i + 15]) for i in idx], np.float32)
        k = np.exp(-0.5 * (np.arange(-10, 11) / 3.4) ** 2)
        prof = np.convolve(np.pad(prof, 10, mode="edge"), k / k.sum(), mode="valid") + 9
        outer = [clock_pt(t, r_) for t, r_ in zip(thetas, prof)]
        tail = [(940, 1070), (860, 1061), (780, 1057), (716, 1064), (692, 1012), (694, 955),
                (710, 900), (745, 850), (783, 800), (805, 750), (825, 700), (842, 655), (862, 611),
                (891, 579), (916, 566)]                           # Unterkante, dann die Arm-Kante
        pts = np.array([(947, 268), (950, 238)] + outer + tail, np.float32)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0], np.cumsum(seg)])
        s_ = np.arange(0, cum[-1], 6.0)
        k = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
        k /= k.sum()
        sm = [np.convolve(np.pad(np.interp(s_, cum, pts[:, c]), 9, mode="edge"), k, mode="valid")
              for c in (0, 1)]
        pts = np.stack(sm, 1)
        self.gold_pts = pts
        n = GOLD_F1 - GOLD_F0 + 1
        self.gold = [render_stroke(pts, 8.5, 1.0, 600, self.tooth, wobble=1.6, taper=(0.01, 0.02),
                                   grain=0.55, upto=(j + 1) / n) for j in range(n)]
        self.sparkle = np.clip(blur(snoise((H, W), 0.5, 12), 0.6), -1, 3)
        self.sparkle /= self.sparkle.std()

    # ............................................................ ein Bild
    def erase_amount(self, f):
        """Wie weit ist die Vorlage an jedem Punkt ausradiert (0…1)?"""
        tex = 0.75 + 0.5 * self.tooth
        e = np.where(f >= self.tpass, np.clip(0.55 + 0.45 * (f - self.tpass) / 2.0, 0, 1), 0) * self.wR
        e = np.clip(e * np.where(e < 0.99, tex, 1), 0, 1)
        ea = np.where(f >= self.tarm, np.clip(0.4 + 0.6 * (f - self.tarm) / 3.0, 0, 1), 0) * self.arm
        ea = np.clip(ea * np.where(ea < 0.99, tex, 1), 0, 1)
        e = np.clip(e + ea * (1 - self.wR), 0, 1)
        # Radierspuren dort, wo am Ende die Uhr weggerieben wird
        for k, (light, full, pts) in enumerate(self.marks):
            fe = MARK_ERASE[k]
            if f >= fe:
                rub = np.zeros((H, W), np.float32)
                full.add(rub, 1.0)
                e = np.maximum(e, np.clip(blur(rub, 13) * 5.5, 0, 0.5) * (1.0 if f > fe else 0.6))
        return (e * (1 - self.goldmask) * self.near).astype(np.float32)

    def render(self, f, noise_seed=None):
        e = self.erase_amount(f)[..., None]
        od = (1 - e) * self.OD + e * (self.ghostOD + self.grain)
        char = np.zeros((H, W), np.float32)
        red = np.zeros((H, W), np.float32)
        redg = np.zeros((H, W), np.float32)
        gold = np.zeros((H, W), np.float32)

        # Zifferblatt
        for k, (light, full, pts) in enumerate(self.marks):
            fs, fe = MARK_START[k], MARK_ERASE[k]
            if f < fs:
                continue
            if f == fs:
                light.add(char, 1.0)
            elif f < fe:
                full.add(char, 1.0)
            elif f == fe:
                full.add(char, 0.5, ghost=True)
                full.add(char, 0.12)
            else:
                full.add(char, 0.30, ghost=True)
        if PIVOT_F <= f < PIVOT_ERASE_F:
            self.pivot[0 if f == PIVOT_F else 1].add(char, 1.0)
        elif f >= PIVOT_ERASE_F:
            self.pivot[1].add(char, 0.28, ghost=True)

        # Zeiger: aktuelle Stellung voll, alle früheren als Geisterspur
        if f >= HAND_GROW[0]:
            if f < HAND_GROW[-1]:
                self.hand_grow[f - HAND_GROW[0]].add(red, 1.0)
            else:
                pos = sum(1 for t in TICKS if f >= t)
                for j in range(pos):
                    age = f - TICKS[j]
                    self.hands[j].add(redg, 0.30 - 0.10 * min(age / 20.0, 1.0), ghost=True)
                if f < HAND_ERASE_F:
                    self.hands[pos].add(red, 1.0)
                elif f == HAND_ERASE_F:
                    self.hands[pos].add(red, 0.3)
                    self.hands[pos].add(redg, 0.4, ghost=True)
                else:
                    self.hands[pos].add(redg, 0.26, ghost=True)

        # Wind in den Haaren
        for fs, grow, end, stages in self.wind:
            if f < fs:
                continue
            for mk in stages[min(f - fs, grow - 1)]:
                if f < end:
                    mk.add(char, 1.0)
                elif f == end:
                    mk.add(char, 0.5, ghost=True)
                else:
                    mk.add(char, 0.24, ghost=True)

        # Gold
        if f >= GOLD_F0:
            self.gold[min(f - GOLD_F0, len(self.gold) - 1)].add(gold, 1.0)

        od = od + (np.clip(char, 0, 1.25) * 1.45)[..., None] * np.array([1.0, 1.0, 0.98], np.float32)
        img = self.P * np.exp(-od)
        light = self.P / self.P.mean()
        a = np.clip(redg, 0, 0.36)[..., None]
        red_col = np.array([0.60, 0.17, 0.12], np.float32)
        img = img * (1 - a * 0.9) + (red_col * light) * a * 0.9 * np.exp(-0.35 * od)
        a = np.clip(red, 0, 1)[..., None] * 0.93
        img = img * (1 - a) + (red_col * light) * a
        a = np.clip(gold, 0, 1)[..., None] * 0.95
        gcol = np.array([0.62, 0.47, 0.20], np.float32) * (1 + 0.10 * self.sparkle[..., None])
        img = img * (1 - a) + (gcol * light) * a

        # Kamera: minimales Flackern und Rauschen wie bei Einzelfotos
        r = np.random.default_rng(1000 + f if noise_seed is None else noise_seed)
        if f > 0:
            img = img * (1 + r.normal(0, 0.004))
            img = img + r.normal(0, 0.0045, (H, W, 1)).astype(np.float32)
        return np.clip(img, 0, 1)


def save(img, path, q=95):
    cv2.imwrite(path, (img[:, :, ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])


# -------------------------------------------------------------------- Ton --
SR = 44100


def _band(n, lo, hi, seed, tilt=0.0):
    """Rauschen, auf ein Frequenzband begrenzt."""
    r = np.random.default_rng(seed)
    spec = np.fft.rfft(r.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1 / SR)
    wgt = sstep(lo * 0.7, lo, fr) * (1 - sstep(hi, hi * 1.4, fr)) / np.maximum(fr, 20) ** tilt
    y = np.fft.irfft(spec * wgt, n)
    return y / (np.abs(y).max() + 1e-9)


def _env(points, n):
    t = np.arange(n) / SR
    return np.interp(t, [p[0] for p in points], [p[1] for p in points])


def make_audio(path):
    n = int(N_FRAMES / FPS * SR)
    t = np.arange(n) / SR
    L, R = np.zeros(n), np.zeros(n)
    ft = lambda f: f / FPS                                         # Bild -> Sekunde
    t_stop, t_gold = ft(STOP_F), ft(GOLD_F0)

    def put(sig, at, gain=1.0, pan=0.5):
        i = int(at * SR)
        m = min(len(sig), n - i)
        if m > 0:
            L[i:i + m] += sig[:m] * gain * math.cos(pan * math.pi / 2)
            R[i:i + m] += sig[:m] * gain * math.sin(pan * math.pi / 2)

    # Wind: schwillt mit der Beschleunigung an, legt sich nach dem Stillstand
    wenv = _env([(0, 0.10), (ft(TICK0), 0.13), (ft(TICKS[12]), 0.32), (t_stop, 0.8), (t_stop + 0.15, 0.7),
                 (t_stop + 1.8, 0.2), (t_gold, 0.11), (n / SR, 0.07)], n)
    for ch, sd in ((L, 1), (R, 2)):
        gust = 0.75 + 0.25 * np.interp(t, np.linspace(0, n / SR, 40),
                                       np.random.default_rng(30 + sd).uniform(-1, 1, 40))
        ch += 0.42 * wenv * gust * _band(n, 90, 900, sd, tilt=0.6)
        ch += 0.16 * wenv ** 2 * gust * _band(n, 900, 4200, sd + 5, tilt=0.3)

    # Ticken des Zeigers
    def tick(k):
        m = int(0.16 * SR)
        tt = np.arange(m) / SR
        det = 1.0 if k % 2 == 0 else 0.84
        y = np.zeros(m)
        for fq, tau, a in ((830, 0.030, 0.55), (1740, 0.018, 0.4), (3150, 0.008, 0.3)):
            y += a * np.sin(2 * math.pi * fq * det * tt) * np.exp(-tt / tau)
        y += 0.6 * _band(m, 2200, 9000, 50 + k) * np.exp(-tt / 0.004)
        return y

    for k, f in enumerate(TICKS):
        put(tick(k), ft(f), gain=0.30 + 0.25 * k / 29, pan=0.55)

    # Kohle (Striche des Zifferblatts, Zeiger) und Radiergummi
    def scratch(dur, seed, lo=1300, hi=5500):
        m = int(dur * SR)
        tt = np.arange(m) / SR
        rough = np.abs(_band(m, 40, 260, seed + 1)) ** 0.7
        e = sstep(0, 0.025, tt) * (1 - sstep(dur - 0.07, dur, tt))
        return _band(m, lo, hi, seed, tilt=0.2) * rough * e

    def rub(reps, seed):
        m = int((0.11 * reps + 0.05) * SR)
        tt = np.arange(m) / SR
        e = np.sin(math.pi * tt / 0.11) ** 2 * (tt < 0.11 * reps)
        return _band(m, 260, 1500, seed, tilt=0.4) * e * (0.7 + 0.3 * np.abs(_band(m, 30, 120, seed + 1)))

    for k, f in enumerate(MARK_START):
        put(scratch(0.17, 100 + k), ft(f) - 0.03, gain=0.42, pan=0.3 + 0.4 * math.sin(k * 0.52) ** 2)
    put(scratch(0.1, 130), ft(PIVOT_F), gain=0.3)
    put(scratch(0.42, 131, 1500, 6500), ft(HAND_GROW[0]) - 0.03, gain=0.38)
    for i in range(6):
        put(rub(3, 200 + i), ft(ARM_F0) + i * 0.2, gain=0.20, pan=0.42)
    put(rub(3, 220), ft(HAND_ERASE_F) - 0.05, gain=0.34)
    for k, f in MARK_ERASE.items():
        put(rub(2, 230 + k), ft(f) - 0.05, gain=0.32, pan=0.3 + 0.4 * math.sin(k * 0.52) ** 2)
    for i in range(int((GOLD_F1 - GOLD_F0) / 3)):
        put(scratch(0.2, 300 + i, 2200, 7000), ft(GOLD_F0) + i * 0.3, gain=0.10, pan=0.6)

    # Musik: tiefer Bordun, wachsende Reibung, nach dem Stillstand ein ruhiger Klang
    def tone(fq, env, detune=0.0, harm=0.25):
        ph = 2 * math.pi * fq * t
        return env * (np.sin(ph) + harm * np.sin(2 * ph + 0.5))  # leicht obertönig

    drone = _env([(0, 0), (1.2, 0.05), (ft(TICK0), 0.07), (t_stop, 0.15), (t_stop + 0.5, 0.05),
                  (t_gold, 0.05), (n / SR - 1.5, 0.06), (n / SR, 0.0)], n)
    tens = _env([(0, 0), (ft(TICK0), 0.0), (ft(TICKS[14]), 0.05), (t_stop, 0.13), (t_stop + 0.12, 0.0),
                 (n / SR, 0)], n)
    warm = _env([(0, 0), (t_gold - 0.3, 0), (t_gold + 2.2, 0.085), (n / SR - 1.6, 0.085), (n / SR, 0)], n)
    mus_l = tone(73.42, drone) + tone(110.0, drone * 0.7) + tone(155.56, tens) + tone(233.08, tens * 0.6)
    mus_r = tone(73.62, drone) + tone(110.3, drone * 0.7) + tone(156.1, tens) + tone(232.5, tens * 0.6)
    for fq, a in ((146.83, 1.0), (220.0, 0.8), (369.99, 0.55), (440.0, 0.3)):
        mus_l += tone(fq, warm * a, harm=0.12)
        mus_r += tone(fq * 1.002, warm * a, harm=0.12)
    L += mus_l
    R += mus_r

    # etwas Raum, Ein-/Ausblenden, sanft begrenzen
    ir = np.random.default_rng(9).standard_normal(int(0.5 * SR)) * np.exp(-np.arange(int(0.5 * SR)) / (0.11 * SR))
    ir /= np.abs(ir).sum() / 6
    out = []
    for ch in (L, R):
        wet = np.fft.irfft(np.fft.rfft(ch, n + len(ir)) * np.fft.rfft(ir, n + len(ir)))[:n]
        y = ch + 0.16 * wet
        y *= sstep(0, 0.25, t) * (1 - sstep(n / SR - 0.7, n / SR, t))
        out.append(np.tanh(y * 1.5) / 1.5)
    st = np.stack(out, 1)
    st = st / np.abs(st).max() * 0.89
    with wave.open(path, "wb") as wv:
        wv.setnchannels(2)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes((st * 32767).astype("<i2").tobytes())


def main():
    film = Film()
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        os.makedirs(os.path.join(OUT, "preview"), exist_ok=True)
        for f in map(int, sys.argv[2:]):
            save(film.render(f), os.path.join(OUT, "preview", f"f{f:03d}.jpg"), 90)
        return
    fd = os.path.join(OUT, "frames")
    os.makedirs(fd, exist_ok=True)
    for f in range(N_FRAMES):
        cv2.imwrite(os.path.join(fd, f"{f:04d}.png"), (film.render(f)[:, :, ::-1] * 255 + 0.5).astype(np.uint8))
        if f % 30 == 0:
            print("Bild", f, flush=True)
    wav = os.path.join(OUT, "ton.wav")
    make_audio(wav)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(fd, "%04d.png"), "-i", wav,
                    "-vf", "scale=1440:1080:flags=lanczos,format=yuv420p", "-r", "30",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "19",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    os.path.join(HERE, "film.mp4")], check=True)
    print("fertig:", os.path.join(HERE, "film.mp4"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ton":
        os.makedirs(OUT, exist_ok=True)
        make_audio(os.path.join(OUT, "ton.wav"))
    else:
        main()
