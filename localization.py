"""
Mobile robot localization: EKF vs. particle filter vs. dead reckoning.

A differential-drive robot drives loops in a 20 m x 20 m field with known landmarks.
It has noisy wheel odometry and a range-bearing sensor (6 m range).

Experiment 1 (tracking): how accurate is each estimator?
Experiment 2 (kidnapped robot): halfway through, the robot is picked up and moved.
  Can each estimator notice and recover?

Run:  python localization.py           (prints results, saves localization.png)
      python localization.py --seed 3 --particles 1000
"""
import argparse, time
import numpy as np

DT, STEPS = 0.1, 600                        # 60 s at 10 Hz
SIG_R, SIG_B = 0.10, np.deg2rad(2.0)        # sensor noise: 10 cm range, 2 deg bearing
MAX_RANGE = 6.0
ALPHA = (0.05, 0.002, 0.002, 0.05)          # odometry noise model (Thrun et al., Probabilistic Robotics)
FIELD = 20.0

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def motion(x, v, w):
    """Unicycle model, works on one pose (3,) or many poses (N,3)."""
    x = np.array(x, dtype=float)
    th = x[..., 2]
    out = x.copy()
    out[..., 0] += v * DT * np.cos(th + w * DT / 2)
    out[..., 1] += v * DT * np.sin(th + w * DT / 2)
    out[..., 2] = wrap(th + w * DT)
    return out

def noisy_control(v, w, rng, n=None):
    sv = np.sqrt(ALPHA[0] * v ** 2 + ALPHA[1] * w ** 2)
    sw = np.sqrt(ALPHA[2] * v ** 2 + ALPHA[3] * w ** 2)
    return v + rng.normal(0, sv, n), w + rng.normal(0, sw, n)

def sense(x, landmarks, rng):
    z = []
    for i, (lx, ly) in enumerate(landmarks):
        dx, dy = lx - x[0], ly - x[1]
        r = np.hypot(dx, dy)
        if r < MAX_RANGE:
            z.append((i, r + rng.normal(0, SIG_R), wrap(np.arctan2(dy, dx) - x[2] + rng.normal(0, SIG_B))))
    return z

# ---------------------------------------------------------------- EKF
class EKF:
    def __init__(self, x0, P0):
        self.x, self.P = np.array(x0, float), np.array(P0, float)
        self.rejected = 0

    def predict(self, v, w):
        th = self.x[2]
        G = np.array([[1, 0, -v * DT * np.sin(th + w * DT / 2)],
                      [0, 1,  v * DT * np.cos(th + w * DT / 2)],
                      [0, 0, 1]])
        V = np.array([[DT * np.cos(th), 0], [DT * np.sin(th), 0], [0, DT]])
        M = np.diag([ALPHA[0] * v ** 2 + ALPHA[1] * w ** 2, ALPHA[2] * v ** 2 + ALPHA[3] * w ** 2])
        self.x = motion(self.x, v, w)
        self.P = G @ self.P @ G.T + V @ M @ V.T

    def update(self, z, landmarks):
        Q = np.diag([SIG_R ** 2, SIG_B ** 2])
        for i, r, b in z:
            dx, dy = landmarks[i] - self.x[:2]
            q = dx * dx + dy * dy
            zhat = np.array([np.sqrt(q), wrap(np.arctan2(dy, dx) - self.x[2])])
            H = np.array([[-dx / np.sqrt(q), -dy / np.sqrt(q), 0],
                          [dy / q, -dx / q, -1]])
            S = H @ self.P @ H.T + Q
            nu = np.array([r - zhat[0], wrap(b - zhat[1])])
            if nu @ np.linalg.solve(S, nu) > 9.21:        # chi-square 99% gate: reject outliers
                self.rejected += 1
                continue
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ nu
            self.x[2] = wrap(self.x[2])
            self.P = (np.eye(3) - K @ H) @ self.P

# ---------------------------------------------------------------- particle filter
class ParticleFilter:
    def __init__(self, x0, n, rng, spread=(0.3, 0.3, 0.1), inject=True):
        self.rng, self.n, self.inject = rng, n, inject
        self.p = np.array(x0) + rng.normal(0, spread, (n, 3))
        self.w = np.ones(n) / n
        self.w_slow = self.w_fast = None            # Augmented MCL averages
        self.est = np.array(x0, float)

    def predict(self, v, w):
        self.est = motion(self.est, v, w)
        vs, ws = noisy_control(v, w, self.rng, self.n)
        th = self.p[:, 2]
        self.p[:, 0] += vs * DT * np.cos(th + ws * DT / 2)
        self.p[:, 1] += vs * DT * np.sin(th + ws * DT / 2)
        self.p[:, 2] = wrap(th + ws * DT)

    def update(self, z, landmarks):
        if not z: return
        logw = np.zeros(self.n)
        for i, r, b in z:
            dx = landmarks[i][0] - self.p[:, 0]
            dy = landmarks[i][1] - self.p[:, 1]
            rh = np.hypot(dx, dy)
            bh = wrap(np.arctan2(dy, dx) - self.p[:, 2])
            # likelihood uses 1.5x the sensor noise so the filter does not collapse onto too few particles
            logw += -0.5 * ((r - rh) / (1.5 * SIG_R)) ** 2 - 0.5 * (wrap(b - bh) / (1.5 * SIG_B)) ** 2
        # average likelihood (relative to a perfect match) tells us if we are lost
        avg = np.mean(np.exp(logw / len(z)))
        self.w = np.exp(logw - logw.max()); self.w /= self.w.sum()
        if self.w_slow is None: self.w_slow = self.w_fast = avg
        self.w_slow += 0.05 * (avg - self.w_slow)
        self.w_fast += 0.5 * (avg - self.w_fast)
        self.est = self.weighted_mean()             # estimate before any random particles are added
        self.resample()

    def resample(self):
        neff = 1.0 / np.sum(self.w ** 2)
        if neff > self.n / 2: return
        idx = np.searchsorted(np.cumsum(self.w), (self.rng.random() + np.arange(self.n)) / self.n)
        self.p = self.p[np.minimum(idx, self.n - 1)]
        self.w = np.ones(self.n) / self.n
        if self.inject:                                # Augmented MCL: add random particles when lost
            ratio = self.w_fast / max(self.w_slow, 1e-12)
            k = int(0.2 * self.n) if ratio < 0.3 else 0   # only when the fit got much worse than usual
            if k:
                j = self.rng.choice(self.n, k, replace=False)
                self.p[j] = np.column_stack([self.rng.uniform(-FIELD / 2, FIELD / 2, (k, 2)),
                                             self.rng.uniform(-np.pi, np.pi, k)])

    def estimate(self):
        return self.est

    def weighted_mean(self):
        m = np.average(self.p[:, :2], axis=0, weights=self.w)
        th = np.arctan2(np.average(np.sin(self.p[:, 2]), weights=self.w),
                        np.average(np.cos(self.p[:, 2]), weights=self.w))
        return np.array([m[0], m[1], th])

# ---------------------------------------------------------------- experiment
def run(rng, landmarks, n_particles, kidnap_at=None, inject=True):
    x = np.array([0.0, -5.0, 0.0])
    dr = x.copy()
    ekf = EKF(x, np.diag([0.1, 0.1, 0.05]) ** 2)
    pf = ParticleFilter(x, n_particles, rng, inject=inject)
    hist = {k: [] for k in ("true", "dr", "ekf", "pf", "ekf_P")}
    t_ekf = t_pf = 0.0
    for k in range(STEPS):
        t = k * DT
        v, w = 1.0, 0.2 + 0.15 * np.sin(0.15 * t)        # wandering loops
        x = motion(x, *noisy_control(v, w, rng))        # true robot (with real wheel slip)
        if kidnap_at is not None and k == kidnap_at:
            x = np.array([x[0] + 5.0, x[1] + 4.0, wrap(x[2] + np.pi / 2)])   # picked up and moved
        z = sense(x, landmarks, rng)
        dr = motion(dr, v, w)                           # dead reckoning: just integrate commands
        s = time.perf_counter(); ekf.predict(v, w); ekf.update(z, landmarks); t_ekf += time.perf_counter() - s
        s = time.perf_counter(); pf.predict(v, w); pf.update(z, landmarks); t_pf += time.perf_counter() - s
        hist["true"].append(x); hist["dr"].append(dr); hist["ekf"].append(ekf.x.copy())
        hist["pf"].append(pf.estimate()); hist["ekf_P"].append(ekf.P.copy())
    hist = {k: np.array(v) for k, v in hist.items()}
    hist["ms"] = (1000 * t_ekf / STEPS, 1000 * t_pf / STEPS)
    hist["rejected"] = ekf.rejected
    return hist

def errors(h, name, sl=slice(None)):
    e = h[name][sl, :2] - h["true"][sl, :2]
    eth = wrap(h[name][sl, 2] - h["true"][sl, 2])
    return np.sqrt(np.mean(np.sum(e ** 2, 1))), np.rad2deg(np.sqrt(np.mean(eth ** 2)))

def recovery_time(h, name, start, tol=0.5, hold=20):
    err = np.linalg.norm(h[name][:, :2] - h["true"][:, :2], axis=1)
    for k in range(start, STEPS - hold):
        if np.all(err[k:k + hold] < tol): return (k - start) * DT
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", type=int, default=500)
    ap.add_argument("--runs", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    landmarks = rng.uniform(-8, 8, (12, 2))

    # Experiment 1: tracking accuracy, averaged over several runs
    res = {k: [] for k in ("dr", "ekf", "pf")}
    ms = []
    for _ in range(a.runs):
        h = run(rng, landmarks, a.particles)
        for k in res: res[k].append(errors(h, k))
        ms.append(h["ms"])
        if not "demo" in locals(): demo = h
    print(f"Experiment 1: tracking, {a.runs} runs of 60 s, {len(landmarks)} landmarks, {a.particles} particles")
    for k, label in (("dr", "Dead reckoning"), ("ekf", "EKF"), ("pf", "Particle filter")):
        p, th = np.mean(res[k], axis=0)
        print(f"  {label:<16} position RMSE {p:5.2f} m   heading RMSE {th:5.1f} deg")
    ems, pms = np.mean(ms, axis=0)
    print(f"  compute per step: EKF {ems:.3f} ms, particle filter {pms:.2f} ms")

    # Experiment 2: kidnapped robot at t = 30 s
    K = STEPS // 2
    rows = []
    for _ in range(a.runs):
        h = run(rng, landmarks, a.particles, kidnap_at=K)
        h2 = run(rng, landmarks, a.particles, kidnap_at=K, inject=False)
        rows.append((recovery_time(h, "ekf", K), recovery_time(h, "pf", K), recovery_time(h2, "pf", K)))
        if "kid" not in locals(): kid = h
    def summary(col):
        vals = [r[col] for r in rows]
        ok = [v for v in vals if v is not None]
        return f"recovered in {len(ok)}/{len(vals)} runs" + (f", median {np.median(ok):.1f} s" if ok else "")
    print("\nExperiment 2: kidnapped robot (moved ~6.4 m and turned 90 deg at t = 30 s)")
    print(f"  EKF (with outlier gating)               {summary(0)}")
    print(f"  Particle filter + random injection      {summary(1)}")
    print(f"  Particle filter without injection       {summary(2)}")
    plot(demo, kid, landmarks, K)

def plot(h, kid, lm, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axs[0]
    ax.plot(*lm.T, "k*", ms=10, label="landmarks")
    for k, st, lab in (("true", "k-", "true"), ("dr", "r:", "dead reckoning"), ("ekf", "b-", "EKF"), ("pf", "g--", "particle filter")):
        ax.plot(h[k][:, 0], h[k][:, 1], st, lw=1.5, label=lab)
    for k in range(0, STEPS, 60):                     # EKF 2-sigma ellipses
        vals, vecs = np.linalg.eigh(h["ekf_P"][k][:2, :2])
        t = np.linspace(0, 2 * np.pi, 40)
        e = vecs @ (2 * np.sqrt(np.maximum(vals, 0))[:, None] * np.array([np.cos(t), np.sin(t)]))
        ax.plot(h["ekf"][k, 0] + e[0], h["ekf"][k, 1] + e[1], "b", lw=.8, alpha=.6)
    ax.set_aspect("equal"); ax.set_title("Tracking (blue ellipses: EKF 2-sigma)"); ax.legend(fontsize=7)
    ax = axs[1]
    tt = np.arange(STEPS) * DT
    for k, st, lab in (("dr", "r:", "dead reckoning"), ("ekf", "b-", "EKF"), ("pf", "g--", "particle filter")):
        ax.plot(tt, np.linalg.norm(h[k][:, :2] - h["true"][:, :2], axis=1), st, label=lab)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)"); ax.set_title("Tracking error"); ax.legend(fontsize=8)
    ax = axs[2]
    for k, st, lab in (("ekf", "b-", "EKF"), ("pf", "g--", "particle filter + injection")):
        ax.plot(tt, np.linalg.norm(kid[k][:, :2] - kid["true"][:, :2], axis=1), st, label=lab)
    ax.axvline(K * DT, color="k", ls=":", label="kidnapped")
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)"); ax.set_title("Kidnapped robot"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("localization.png", dpi=120)
    print("\nsaved localization.png")

if __name__ == "__main__":
    main()
