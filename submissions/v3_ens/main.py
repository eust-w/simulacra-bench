"""SimulacraBench submission: multi-task nets + count backoff, stacked by CV.

Held-out respondents show only their GIVEN block, so every scored cell is a
draw from p(item | GIVEN). Everything here is fitted inside predict() on the
visible rows, and every choice that could overfit is made by K-fold
cross-validation on those rows, never tuned on the sandbox:

  M0  crowd marginal (counts + 1/2)                 -- the zero of the metric
  M1  Dirichlet backoff over the 2 most informative GIVEN columns
  M2  multi-task softmax regression on one-hot GIVEN
  M3  multi-task MLP (shared hidden layers, one softmax head per item)

M2/M3 keep, per item, the epoch with the best validation loss on their fold,
and the hidden rows get the average of the K fold models. The out-of-fold
probabilities of all four are blended per item by a convex mixture fitted to
the out-of-fold log score (with the grader's floor applied), shrunk towards
one shared mixture. Finally, gate rules that hold without exception on the
visible rows (parent in GIVEN, or a chain closed by one) are applied exactly.
"""

import os
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - the hosted image ships torch
    torch = None

FLOOR = 1e-3
FOLDS = int(os.environ.get("SB_FOLDS", "5"))
SEED = 0
BUDGET = float(os.environ.get("SB_BUDGET", "240"))   # soft seconds per call
TOTAL = float(os.environ.get("SB_TOTAL", "650"))     # soft seconds since import
T_START = time.time()

if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
else:
    DEVICE = None


# ------------------------------------------------------------- utilities --

def _codes(series, levels):
    lookup = {level: i for i, level in enumerate(levels)}
    return np.array(series.map(lookup).fillna(-1).astype(np.int64).to_numpy())


def _folds(n, k, seed):
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


def _true_prob(probs, y):
    """probs (N, J, Kmax), y (N, J) -> (N, J) probability of the answer."""
    return np.take_along_axis(probs, y[:, :, None], axis=2)[:, :, 0]


# ------------------------------------------------------------ M0 and M1 --

def marginal(y, widths, kmax):
    out = np.zeros((len(widths), kmax))
    for j, w in enumerate(widths):
        c = np.bincount(y[:, j], minlength=w)[:w] + 0.5
        out[j, :w] = c / c.sum()
    return out


def mutual_information(x, y, nx, ny):
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def backoff(xg_fit, y_fit, xg_all, cards, widths, kmax, alpha=2.0, strata=2):
    """Dirichlet backoff: marginal -> 1 column -> 2 columns (by MI)."""
    n = xg_all.shape[0]
    out = np.zeros((n, len(widths), kmax))
    for j, w in enumerate(widths):
        yj = y_fit[:, j]
        mi = [mutual_information(xg_fit[:, g], yj, cards[g], w)
              for g in range(len(cards))]
        ladder = list(np.argsort(mi)[::-1][:strata])
        c0 = np.bincount(yj, minlength=w)[:w] + 0.5
        running = np.tile(c0 / c0.sum(), (n, 1))
        key_fit = np.zeros(len(yj), np.int64)
        key_all = np.zeros(n, np.int64)
        size = 1
        for g in ladder:
            key_fit = key_fit * cards[g] + xg_fit[:, g]
            key_all = key_all * cards[g] + xg_all[:, g]
            size *= cards[g]
            table = np.bincount(key_fit * w + yj, minlength=size * w)
            counts = table.reshape(size, w)[key_all].astype(float)
            running = (counts + alpha * running) / (
                counts.sum(axis=1, keepdims=True) + alpha)
        out[:, j, :w] = running
    return out


# ------------------------------------------------------------ M2 and M3 --

class Net(torch.nn.Module if torch is not None else object):
    def __init__(self, d_in, n_items, kmax, hidden, dropout):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.GELU(),
                       torch.nn.Dropout(dropout)]
            d = h
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, n_items * kmax)
        self.shape = (n_items, kmax)

    def forward(self, x):
        return self.head(self.body(x)).view(-1, *self.shape)


def fit_net(x_tr, y_tr, x_va, y_va, x_te, mask, cfg, seed):
    """Train one net; per item keep the epoch with the best validation loss.

    Returns probabilities on the validation rows and on the test rows, both
    (N, J, Kmax), each item taken from its own best epoch.
    """
    torch.manual_seed(seed)
    n_items, kmax = mask.shape
    net = Net(x_tr.shape[1], n_items, kmax, cfg["hidden"], cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    neg = (~mask).to(DEVICE) * -1e9
    xt, yt = x_tr.to(DEVICE), y_tr.to(DEVICE)
    xv, yv, xe = x_va.to(DEVICE), y_va.to(DEVICE), x_te.to(DEVICE)
    n = xt.shape[0]
    bs = cfg["batch"]
    best = torch.full((n_items,), float("inf"), device=DEVICE)
    best_va = torch.zeros(xv.shape[0], n_items, kmax, device=DEVICE)
    best_te = torch.zeros(xe.shape[0], n_items, kmax, device=DEVICE)
    stale = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logp = F.log_softmax(net(xt[idx]) + neg, dim=2)
            loss = -logp.gather(2, yt[idx][:, :, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv = F.log_softmax(net(xv) + neg, dim=2)
            per_item = -lv.gather(2, yv[:, :, None])[:, :, 0].mean(0)
            better = per_item < best - 1e-5
            if better.any():
                pe = F.softmax(net(xe) + neg, dim=2)
                best_va[:, better] = lv.exp()[:, better]
                best_te[:, better] = pe[:, better]
                best = torch.where(better, per_item, best)
                stale = 0
            else:
                stale += 1
        if stale >= cfg["patience"]:
            break
    return best_va.cpu().numpy(), best_te.cpu().numpy()


NET_CONFIGS = {
    "linear": dict(hidden=[], dropout=0.0, lr=1e-2, wd=1e-4, batch=512,
                   epochs=60, patience=6),
    "mlp": dict(hidden=[256, 256], dropout=0.3, lr=2e-3, wd=1e-4, batch=256,
                epochs=60, patience=6),
    "mlp_wide": dict(hidden=[512], dropout=0.5, lr=2e-3, wd=1e-3, batch=256,
                     epochs=60, patience=6),
    "mlp_deep": dict(hidden=[256, 128, 128], dropout=0.2, lr=1e-3, wd=1e-4,
                     batch=256, epochs=80, patience=8),
}


# -------------------------------------------------------------- stacking --

def fit_blend(ptrue, widths, lam=0.05, steps=400):
    """Per-item convex weights over M models maximizing the floored OOF score.

    ptrue (M, N, J): out-of-fold probability each model gave the true answer.
    Weights are softmax(B + A_j); A_j is penalized, so items with little
    evidence fall back on the shared mixture B.
    """
    m, n, j = ptrue.shape
    p = torch.tensor(ptrue, dtype=torch.float64)
    k = torch.tensor(widths, dtype=torch.float64)
    a = torch.zeros(j, m, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(m, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.05)
    for _ in range(steps):
        w = torch.softmax(a + b, dim=1)                   # (J, M)
        mix = (p * w.T[:, None, :]).sum(0)                # (N, J)
        mix = FLOOR + (1 - k * FLOOR) * mix
        loss = -torch.log(mix).mean() + lam * (a ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(a + b, dim=1).detach().numpy()


def blend_numpy(ptrue, widths):
    """Fallback without torch: one shared weight vector by grid search."""
    k = np.asarray(widths, float)
    best, best_w = -np.inf, None
    m = ptrue.shape[0]
    for w0 in np.linspace(0, 1, 11):
        w = np.zeros(m)
        w[0], w[1] = w0, 1 - w0
        mix = np.tensordot(w, ptrue, axes=1)
        s = np.log(FLOOR + (1 - k * FLOOR) * mix).mean()
        if s > best:
            best, best_w = s, w
    return np.tile(best_w, (ptrue.shape[2], 1))


# ------------------------------------------------------------ gate rules --

def gate_rules(schema, targets, given, frame, visible_mask, options):
    """Rows whose gate is decided by GIVEN alone, verified on visible rows.

    Returns {item: (closed, open)} boolean arrays over all rows, only for items
    where the rule held for every visible respondent.
    """
    items = schema["items"]
    gated_value = schema["gated_value"]
    n = len(frame)
    closed, opened = {}, {}

    def resolve(name, depth=0):
        if name in closed or depth > 50:
            return closed.get(name), opened.get(name)
        gate = items[name].get("gate")
        if not gate:
            return np.zeros(n, bool), np.zeros(n, bool)
        parent = gate["parent"]
        allowed = np.asarray(gate["observed_if"], dtype=object)
        if items[parent]["class"] == "GIVEN":
            pv = frame[parent].to_numpy(dtype=object)
            c = ~np.isin(pv, allowed)
            o = ~c
            # A GIVEN parent that is itself gated closes nothing extra: its
            # value is visible, sentinel included.
        elif parent in targets:
            pc, _ = resolve(parent, depth + 1)
            c = pc.copy() if pc is not None else np.zeros(n, bool)
            o = np.zeros(n, bool)
        else:
            c, o = np.zeros(n, bool), np.zeros(n, bool)
        closed[name], opened[name] = c, o
        return c, o

    rules = {}
    for name in targets:
        if not items[name].get("gate"):
            continue
        c, o = resolve(name)
        if not (c.any() or o.any()):
            continue
        sentinel = frame[name].to_numpy(dtype=object) == gated_value
        vis = visible_mask
        if (sentinel[vis & c].all() and not sentinel[vis & o].any()
                and (vis & (c | o)).sum() >= 20):
            rules[name] = (c, o)
    return rules


# --------------------------------------------------------------- predict --

def predict(frame, schema):
    t0 = time.time()
    records = schema["items"]
    items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if records[n]["class"] == "GIVEN"]
    targets = [n for n in items if records[n]["class"] == "PREDICT"]
    gated_value = schema["gated_value"]
    options = {n: list(records[n]["values"]) + (
        [gated_value] if records[n].get("gate") else []) for n in items}
    widths = [len(options[t]) for t in targets]
    kmax = max(widths)
    n_items = len(targets)

    visible_mask = frame[targets].notna().all(axis=1).to_numpy()
    hidden_mask = frame[targets].isna().any(axis=1).to_numpy()
    vis_idx = np.flatnonzero(visible_mask)
    hid_idx = np.flatnonzero(hidden_mask)

    # GIVEN codes; a level outside the schema (should not happen) -> extra slot
    cards = []
    xg = np.zeros((len(frame), len(given)), np.int64)
    for g_i, g in enumerate(given):
        lv = list(records[g]["values"]) + [gated_value]
        c = _codes(frame[g], lv)
        c[c < 0] = len(lv)
        xg[:, g_i] = c
        cards.append(len(lv) + 1)
    offsets = np.concatenate([[0], np.cumsum(cards)[:-1]])
    d_in = int(sum(cards))

    y = np.zeros((len(vis_idx), n_items), np.int64)
    for j, t in enumerate(targets):
        c = _codes(frame[t].iloc[vis_idx], options[t])
        c[c < 0] = 0
        y[:, j] = c

    n_vis, n_hid = len(vis_idx), len(hid_idx)
    xg_vis, xg_hid = xg[vis_idx], xg[hid_idx]

    mask = np.zeros((n_items, kmax), bool)
    for j, w in enumerate(widths):
        mask[j, :w] = True

    use_torch = torch is not None and n_vis >= 200
    model_names = ["marginal", "backoff"] + (list(NET_CONFIGS) if use_torch else [])
    oof = {m: np.zeros((n_vis, n_items, kmax), np.float32) for m in model_names}
    test = {m: np.zeros((n_hid, n_items, kmax), np.float32) for m in model_names}

    def onehot(rows):
        x = np.zeros((rows.shape[0], d_in), np.float32)
        for g_i in range(rows.shape[1]):
            x[np.arange(rows.shape[0]), offsets[g_i] + rows[:, g_i]] = 1.0
        return torch.tensor(x) if use_torch else x

    folds = _folds(n_vis, FOLDS, SEED) if n_vis >= FOLDS * 20 else None
    if folds is not None:
        x_hid_t = onehot(xg_hid) if use_torch else None
        for f, va in enumerate(folds):
            tr = np.setdiff1d(np.arange(n_vis), va)
            oof["marginal"][va] = marginal(y[tr], widths, kmax)[None]
            oof["backoff"][va] = backoff(xg_vis[tr], y[tr], xg_vis[va], cards,
                                         widths, kmax)
            if use_torch:
                x_tr, x_va = onehot(xg_vis[tr]), onehot(xg_vis[va])
                y_tr, y_va = torch.tensor(y[tr]), torch.tensor(y[va])
                for name, cfg in NET_CONFIGS.items():
                    late = (time.time() - t0 > BUDGET
                            or time.time() - T_START > TOTAL)
                    if late and f > 0:
                        # out of time: reuse the fold models we already have
                        oof[name][va] = oof["backoff"][va]
                        continue
                    pv, pt = fit_net(x_tr, y_tr, x_va, y_va, x_hid_t,
                                     torch.tensor(mask), cfg, SEED + 17 * f)
                    oof[name][va] = pv
                    test[name] += pt
                    test.setdefault("_n_" + name, 0)
                    test["_n_" + name] += 1
        for name in NET_CONFIGS if use_torch else []:
            test[name] /= max(1, test.pop("_n_" + name, 1))

        ptrue = np.stack([_true_prob(oof[m], y) for m in model_names])
        weights = (fit_blend(ptrue, widths) if torch is not None
                   else blend_numpy(ptrue, widths))
        if os.environ.get("SB_DEBUG"):
            k = np.asarray(widths, float)
            for m_i, m in enumerate(model_names):
                s = np.log(FLOOR + (1 - k * FLOOR) * ptrue[m_i]).mean()
                print("  oof %-9s %.4f  mean weight %.3f"
                      % (m, s, weights[:, m_i].mean()))
            mix = (ptrue * weights.T[:, None, :]).sum(0)
            print("  oof blend     %.4f   (%.1fs)" % (
                np.log(FLOOR + (1 - k * FLOOR) * mix).mean(), time.time() - t0))
    else:
        weights = np.tile(np.eye(len(model_names))[1], (n_items, 1))

    test["marginal"][:] = marginal(y, widths, kmax)[None]
    test["backoff"] = backoff(xg_vis, y, xg_hid, cards, widths, kmax).astype(np.float32)
    if folds is None and use_torch:
        for name in NET_CONFIGS:
            test[name] = test["backoff"]

    final = np.zeros((n_hid, n_items, kmax))
    for m_i, m in enumerate(model_names):
        final += weights[None, :, m_i, None] * test[m]

    # Gate rules that held without exception on the visible rows.
    rules = gate_rules(schema, targets, given, frame, visible_mask, options)
    for j, t in enumerate(targets):
        if t not in rules:
            continue
        c, o = rules[t]
        c, o = c[hid_idx], o[hid_idx]
        w = widths[j]
        final[c, j, :w] = 0.0
        final[c, j, w - 1] = 1.0
        final[o, j, w - 1] = 0.0

    final = np.clip(final, 0, None) + 1e-12
    final /= final.sum(axis=2, keepdims=True)

    # One vector per blank cell, rows top to bottom, items in schema order.
    pos = {t: j for j, t in enumerate(targets)}
    grid = frame[items].isna().to_numpy()
    out = []
    hid_row = {r: i for i, r in enumerate(hid_idx)}
    for r in hid_idx:
        row = final[hid_row[r]]
        for col, name in enumerate(items):
            if grid[r, col]:
                j = pos[name]
                out.append(row[j, :widths[j]].tolist())
    return out
