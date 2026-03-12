#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


DEFAULT_PARAMS = dict(
    # Flows (L/min)
    Qli=1.45,
    Qco=5.6,
    fd1=0.692,
    fd2=0.0842,

    # Volumes (L)
    Vli=1.69,
    VliV=0.29,
    VliIF=0.27,
    VliIC=1.13,
    V1=24.3,
    V2=38.4,
    Vvpl=5.2,

    # Partition coefficients
    Rli1=0.39,
    Rli2=0.022,
    R1=0.655,
    R2=0.655,

    # Membrane transport (L/min) — whole liver totals
    h1=27416.0,
    h2=14606.0,

    # Hepatic elimination (Michaelis–Menten) — whole liver total
    Vmax=12.16,   # mg/min (whole liver)
    Km=1.08,      # mg/L
    fup=0.03,

    # Dosing
    IVDose=9.6,          # mg
    BolusDuration=1.0,   # min

    # Time grid
    TSTOP=720.0,     # min
    POINTS=800
)


def iv_smooth_rate(t, dose, bolus_duration):
    if t < 0.0:
        return 0.0
    D = max(float(bolus_duration), 1e-9)
    lam = 1.0 / D
    if t <= D:
        # smooth beta-like pulse shape
        shape = 30.0 * lam * (lam * t**2) * (1.0 - lam * t)**2
        return float(dose) * shape
    return 0.0


def loguniform(low, high, size=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    low = float(low); high = float(high)
    if low <= 0 or high <= 0 or high <= low:
        raise ValueError("loguniform requires 0 < low < high")
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def sample_clusters(n_cells, n_clusters=4, cluster_fracs=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if cluster_fracs is None:
        cluster_fracs = np.full(n_clusters, 1.0 / n_clusters, dtype=float)
    else:
        cluster_fracs = np.asarray(cluster_fracs, float)
        cluster_fracs = cluster_fracs / cluster_fracs.sum()

    counts = np.floor(cluster_fracs * n_cells).astype(int)
    rem = n_cells - counts.sum()
    if rem > 0:
        frac_parts = (cluster_fracs * n_cells) - np.floor(cluster_fracs * n_cells)
        add_idx = np.argsort(-frac_parts)[:rem]
        counts[add_idx] += 1

    cl = np.concatenate([np.full(counts[k], k, dtype=int) for k in range(n_clusters)])
    rng.shuffle(cl)
    return cl


def nb_multiplier_from_counts(n, mu_mult, theta, mu_count=100.0, rng=None, floor=1e-12):

    rng = np.random.default_rng() if rng is None else rng
    mu_mult = float(mu_mult)
    theta = float(theta)
    mu_count = float(mu_count)

    if mu_mult <= 0:
        return np.full(n, floor, dtype=float)
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if mu_count <= 0:
        raise ValueError("mu_count must be > 0")

    lam = rng.gamma(shape=theta, scale=mu_count/theta, size=n)
    k = rng.poisson(lam).astype(float)
    k = np.maximum(k, 1.0)  # keep strictly positive
    w = (k / max(k.mean(), floor)) * mu_mult
    return np.maximum(w, floor)


def sample_clustered_nb_multipliers_optionA(
    cluster_id,
    mus, thetas,
    mu_count=100.0,
    seed=123,
    floor=1e-12
):
    cluster_id = np.asarray(cluster_id, dtype=int)
    mus = np.asarray(mus, float)
    thetas = np.asarray(thetas, float)

    K = int(cluster_id.max()) + 1
    if mus.shape != (K,) or thetas.shape != (K,):
        raise ValueError(f"mus/thetas must be shape ({K},)")

    W = np.empty(cluster_id.size, dtype=float)
    for k in range(K):
        idx = np.where(cluster_id == k)[0]
        if idx.size == 0:
            continue
        rngk = np.random.default_rng(seed + 10007*k)
        W[idx] = nb_multiplier_from_counts(
            n=idx.size,
            mu_mult=mus[k],
            theta=thetas[k],
            mu_count=mu_count,
            rng=rngk,
            floor=floor
        )
    return np.maximum(W, floor)


def rhs_mdz_multicell_with_bulk(
    t, y, p,
    VliIC_i, V_IC_bulk,
    Vmax_i, Vmax_bulk_total,
    h1_cell, h2_cell,
    N_bulk
):
    n_het = len(VliIC_i)

    # indices
    i_AliV      = 0
    i_AliIF     = 1
    i_AliIC_s   = 2
    i_AliIC_e   = 2 + n_het
    i_AIC_bulk  = i_AliIC_e
    i_Av1       = i_AIC_bulk + 1
    i_Av2       = i_AIC_bulk + 2
    i_Avpl      = i_AIC_bulk + 3
    i_AUC_li    = i_AIC_bulk + 4
    i_AUC_vpl   = i_AIC_bulk + 5

    # Unpack states
    AliV      = y[i_AliV]
    AliIF     = y[i_AliIF]
    AliIC_i   = y[i_AliIC_s:i_AliIC_e]
    AIC_bulk  = y[i_AIC_bulk]
    Av1       = y[i_Av1]
    Av2       = y[i_Av2]
    Avpl      = y[i_Avpl]
    AUC_li    = y[i_AUC_li]
    AUC_vpl   = y[i_AUC_vpl]

    # Parameters / geometry
    Qli, Qco, fd1, fd2 = p["Qli"], p["Qco"], p["fd1"], p["fd2"]
    Vli, VliV, VliIF, V1, V2, Vvpl = p["Vli"], p["VliV"], p["VliIF"], p["V1"], p["V2"], p["Vvpl"]
    Rli1, Rli2, R1, R2 = p["Rli1"], p["Rli2"], p["R1"], p["R2"]
    Km, fup = p["Km"], p["fup"]
    IVDose, Dbol = p["IVDose"], p["BolusDuration"]

    # Concentrations
    CliV      = AliV    / VliV
    CliIF     = AliIF   / VliIF
    CliIC_i   = AliIC_i / VliIC_i
    C_IC_bulk = AIC_bulk / V_IC_bulk

    Cv1  = Av1  / V1
    Cv2  = Av2  / V2
    Cvpl = Avpl / Vvpl

    # Lumped tissue flows
    Q_rest = max(Qco - Qli, 0.0)
    Q1 = fd1 * Q_rest
    Q2 = fd2 * Q_rest

    # -------- Liver vascular --------
    rliIn  = Qli * Cvpl
    rliOut = Qli * CliV

    h1_total = p["h1_total"]
    nliVIF = h1_total * (CliV - CliIF / Rli1)
    dAliV = rliIn - rliOut - nliVIF

    # -------- Liver IF ↔ IC --------
    nliIFIC_i = h2_cell * (CliIF - CliIC_i / Rli2)
    flux_explicit = np.sum(nliIFIC_i)

    nliIFIC_bulk_single = h2_cell * (CliIF - C_IC_bulk / Rli2)
    flux_bulk_total = N_bulk * nliIFIC_bulk_single

    dAliIF = nliVIF - (flux_explicit + flux_bulk_total)

    # -------- IC elimination (explicit cells) --------
    Ucon = Rli2 / max(fup, 1e-12)
    elim_i = Vmax_i * CliIC_i / (Km * Ucon + CliIC_i)
    dAliIC_i = nliIFIC_i - elim_i

    # -------- IC elimination (bulk group) --------
    elim_bulk = Vmax_bulk_total * C_IC_bulk / (Km * Ucon + C_IC_bulk)
    dAIC_bulk = flux_bulk_total - elim_bulk

    # -------- Lumped tissues --------
    rv1In  = Q1 * Cvpl
    rv1Out = Q1 * (Cv1 / R1)
    dAv1 = rv1In - rv1Out

    rv2In  = Q2 * Cvpl
    rv2Out = Q2 * (Cv2 / R2)
    dAv2 = rv2In - rv2Out

    # -------- Venous plasma --------
    rvplIn  = rliOut + rv1Out + rv2Out
    rvplOut = rliIn  + rv1In  + rv2In

    iv_rate = iv_smooth_rate(t, IVDose, Dbol)
    dAvpl = rvplIn + iv_rate - rvplOut

    # -------- AUC trackers --------
    AliIC_total_liver = np.sum(AliIC_i) + AIC_bulk
    CliT = (AliV + AliIF + AliIC_total_liver) / Vli
    dAUC_li  = CliT
    dAUC_vpl = Cvpl

    # Pack derivatives
    dydt = np.zeros_like(y)
    dydt[i_AliV] = dAliV
    dydt[i_AliIF] = dAliIF
    dydt[i_AliIC_s:i_AliIC_e] = dAliIC_i
    dydt[i_AIC_bulk] = dAIC_bulk
    dydt[i_Av1] = dAv1
    dydt[i_Av2] = dAv2
    dydt[i_Avpl] = dAvpl
    dydt[i_AUC_li] = dAUC_li
    dydt[i_AUC_vpl] = dAUC_vpl
    return dydt


def simulate_mdz_multicell_with_bulk(
    params=None,
    n_het_cells=2000,
    N_total_cells=2_000_000_000,   # NOTE: 2e9 (fix the earlier 2_000_000_0 typo)
    Vmax_multipliers=None,
    t_eval=None,
    debug=True
):
    p = DEFAULT_PARAMS.copy()
    if params is not None:
        p.update(params)

    # Time grid
    TSTOP  = float(p["TSTOP"])
    POINTS = int(p["POINTS"])
    if t_eval is None:
        t_eval = np.linspace(0.0, TSTOP, POINTS)

    # Cell counts
    n_het  = int(n_het_cells)
    N_tot  = int(N_total_cells)
    if N_tot <= n_het:
        raise ValueError("N_total_cells must be > n_het_cells to have a bulk group.")
    N_bulk = N_tot - n_het

    # Volumes per cell and per group
    VliIC_total = float(p["VliIC"])
    v_cell      = VliIC_total / N_tot
    VliIC_i     = np.full(n_het, v_cell, dtype=float)
    V_IC_bulk   = v_cell * N_bulk

    # Vmax allocation (conserve whole-liver Vmax)
    Vmax_total_liver  = float(p["Vmax"])
    Vmax_per_cell_avg = Vmax_total_liver / N_tot

    mult = np.ones(n_het, dtype=float) if Vmax_multipliers is None else np.asarray(Vmax_multipliers, float)
    if mult.shape != (n_het,):
        raise ValueError(f"Vmax_multipliers must have shape ({n_het},)")
    mult = np.maximum(mult, 1e-12)

    # rescale explicit mean to 1 so totals remain consistent
    mult_rescaled = mult / mult.mean()

    Vmax_i = Vmax_per_cell_avg * mult_rescaled
    Vmax_explicit_total = float(np.sum(Vmax_i))
    Vmax_bulk_total = max(Vmax_total_liver - Vmax_explicit_total, 0.0)

    # Per-cell h1, h2
    h1_total = float(p["h1"])
    h2_total = float(p["h2"])
    h1_cell  = h1_total / N_tot
    h2_cell  = h2_total / N_tot

    # stored so RHS uses full-liver h1
    p["h1_total"] = h1_total

    if debug:
        print("\n--- MDZ multicell liver setup (Option A; with bulk group) ---")
        print(f"n_het_cells   = {n_het}")
        print(f"N_total_cells = {N_tot}")
        print(f"N_bulk        = {N_bulk}")
        print(f"VliIC_total   = {VliIC_total:.4f} L")
        print(f"v_cell        = {v_cell:.4e} L per cell")
        print(f"Sum(VliIC_i)  = {VliIC_i.sum():.4e} L (explicit cells)")
        print(f"V_IC_bulk     = {V_IC_bulk:.4f} L (bulk group)")
        print(f"Vmax_liver_total    = {Vmax_total_liver:.6f} mg/min")
        print(f"Vmax_per_cell_avg   = {Vmax_per_cell_avg:.3e} mg/min per cell")
        print(f"Vmax_explicit_total = {Vmax_explicit_total:.6f} mg/min")
        print(f"Vmax_bulk_total     = {Vmax_bulk_total:.6f} mg/min")
        print(f"Vmax_i min/mean/max = {Vmax_i.min():.3e} / {Vmax_i.mean():.3e} / {Vmax_i.max():.3e}")
        print(f"h1_total={h1_total:.4e}, h1_cell={h1_cell:.4e}")
        print(f"h2_total={h2_total:.4e}, h2_cell={h2_cell:.4e}")

        if h2_cell > 0:
            tau_diff_min = v_cell / h2_cell
            tau_diff_sec = tau_diff_min * 60.0
            print(f"Approx diffusion timescale (IC-IF): {tau_diff_min:.3e} min (~{tau_diff_sec:.3e} s)")
        else:
            print("h2_cell=0 -> infinite diffusion timescale")

        ts = np.linspace(0, p["BolusDuration"], 1001)
        iv = np.array([iv_smooth_rate(tt, p["IVDose"], p["BolusDuration"]) for tt in ts])
        area = np.trapz(iv, ts)
        print(f"IV smooth check: ∫ rate dt = {area:.6f} mg (target {p['IVDose']})")
        print("------------------------------------------------------------")

    # Initial conditions
    n_states = n_het + 8  # AliV, AliIF, AliIC(n_het), AIC_bulk, Av1, Av2, Avpl, AUC_li, AUC_vpl
    y0 = np.zeros(n_states, dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rhs_mdz_multicell_with_bulk(
            t, y, p,
            VliIC_i, V_IC_bulk,
            Vmax_i, Vmax_bulk_total,
            h1_cell, h2_cell,
            N_bulk
        ),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-8
    )

    if debug:
        print("solve_ivp status:", sol.status, "| message:", sol.message)

    # Unpack for outputs
    i_AliV      = 0
    i_AliIF     = 1
    i_AliIC_s   = 2
    i_AliIC_e   = 2 + n_het
    i_AIC_bulk  = i_AliIC_e
    i_Av1       = i_AIC_bulk + 1
    i_Av2       = i_AIC_bulk + 2
    i_Avpl      = i_AIC_bulk + 3
    i_AUC_li    = i_AIC_bulk + 4
    i_AUC_vpl   = i_AIC_bulk + 5

    AliV     = sol.y[i_AliV, :]
    AliIF    = sol.y[i_AliIF, :]
    AliIC    = sol.y[i_AliIC_s:i_AliIC_e, :]
    AIC_bulk = sol.y[i_AIC_bulk, :]
    Av1      = sol.y[i_Av1, :]
    Av2      = sol.y[i_Av2, :]
    Avpl     = sol.y[i_Avpl, :]
    AUC_li   = sol.y[i_AUC_li, :]
    AUC_vpl  = sol.y[i_AUC_vpl, :]

    VliV, VliIF, V1, V2, Vvpl, Vli = p["VliV"], p["VliIF"], p["V1"], p["V2"], p["Vvpl"], p["Vli"]

    CliV      = AliV / VliV
    CliIF     = AliIF / VliIF
    CliIC_mat = AliIC / VliIC_i[:, None]
    C_IC_bulk = AIC_bulk / V_IC_bulk
    Cv1       = Av1 / V1
    Cv2       = Av2 / V2
    Cvpl      = Avpl / Vvpl

    AliIC_total_liver = AliIC.sum(axis=0) + AIC_bulk
    CliT = (AliV + AliIF + AliIC_total_liver) / Vli

    df = pd.DataFrame({
        "Time_min": sol.t,
        "Cvpl_mg_per_L":  Cvpl,
        "Cv1_mg_per_L":   Cv1,
        "Cv2_mg_per_L":   Cv2,
        "CliV_mg_per_L":  CliV,
        "CliIF_mg_per_L": CliIF,
        "CliT_mg_per_L":  CliT,
        "C_IC_bulk_mg_per_L": C_IC_bulk,
        "AUC_liver":      AUC_li,
        "AUC_vpl":        AUC_vpl
    })

    return sol, df, CliIC_mat, C_IC_bulk, VliIC_i, Vmax_i


def pick_indices_balanced_by_cluster(cluster_id, n_keep_per_cluster, seed=123):
    cluster_id = np.asarray(cluster_id, dtype=int)
    K = int(cluster_id.max()) + 1
    rng = np.random.default_rng(seed)
    kept = []
    for k in range(K):
        idx_k = np.where(cluster_id == k)[0]
        if idx_k.size == 0:
            continue
        take = min(n_keep_per_cluster, idx_k.size)
        kept.append(rng.choice(idx_k, size=take, replace=False))
    kept = np.concatenate(kept) if len(kept) else np.array([], dtype=int)
    return kept[np.lexsort((kept, cluster_id[kept]))]


def sample_indices_per_cluster(cluster_id, n_show_per_cluster, seed=1):
    cluster_id = np.asarray(cluster_id, dtype=int)
    K = int(cluster_id.max()) + 1
    rng = np.random.default_rng(seed)
    idx_by_cluster = []
    for k in range(K):
        idx_k = np.where(cluster_id == k)[0]
        if idx_k.size == 0:
            idx_by_cluster.append(np.array([], dtype=int))
            continue
        idx_by_cluster.append(rng.choice(idx_k, size=min(n_show_per_cluster, idx_k.size), replace=False))
    return idx_by_cluster


def plot_basic_system_traces(t, df):
    for ycol, ylabel, title in [
        ("Cvpl_mg_per_L",  "Venous plasma conc (mg/L)", "Venous plasma concentration vs time"),
        ("CliV_mg_per_L",  "Liver vascular conc (mg/L)", "Liver vascular concentration vs time"),
        ("CliIF_mg_per_L", "Liver IF conc (mg/L)", "Liver interstitial fluid concentration vs time"),
    ]:
        plt.figure(figsize=(6, 4))
        plt.plot(t, df[ycol], lw=2)
        plt.xlabel("Time (min)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, ls="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


def plot_per_cell_traces(t, mat, idx_by_cluster, title, ylabel):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(7, 4))
    for k, idx_plot in enumerate(idx_by_cluster):
        for j, i in enumerate(idx_plot):
            plt.plot(t, mat[i, :], alpha=0.35, color=colors[k % len(colors)],
                     label=(f"Cluster {k}" if j == 0 else None))
    plt.xlabel("Time (min)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_cluster_timebands(
    t_minutes,
    Y_mat,              # (n_cells, n_time)
    cluster_id,         # (n_cells,)
    K=4,
    title="Per-cluster time course (mean + IQR + 5–95%)",
    y_label="Value",
    x_label="Time (hr)",
    convert_time_to_hr=True,
    use_subset_per_cluster=None,
    seed=1
):
    t = np.asarray(t_minutes, float)
    if convert_time_to_hr:
        t = t / 60.0

    Y = np.asarray(Y_mat, float)
    cluster_id = np.asarray(cluster_id, int)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    for k in range(K):
        ax = axes[k]
        idx = np.where(cluster_id == k)[0]
        if idx.size == 0:
            ax.set_title(f"Cluster {k} (n=0)")
            ax.axis("off")
            continue

        if use_subset_per_cluster is not None and idx.size > use_subset_per_cluster:
            idx = rng.choice(idx, size=use_subset_per_cluster, replace=False)

        Yk = Y[idx, :]
        q05 = np.quantile(Yk, 0.05, axis=0)
        q25 = np.quantile(Yk, 0.25, axis=0)
        q75 = np.quantile(Yk, 0.75, axis=0)
        q95 = np.quantile(Yk, 0.95, axis=0)
        mean = np.mean(Yk, axis=0)

        ax.fill_between(t, q05, q95, alpha=0.15)
        ax.fill_between(t, q25, q75, alpha=0.30)
        ax.plot(t, mean, lw=3)

        ax.set_title(f"Cluster {k} (n={Yk.shape[0]})")
        ax.grid(True, ls="--", alpha=0.4)

    for ax in axes[2:]:
        ax.set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[2].set_ylabel(y_label)

    handles = [
        Line2D([0], [0], lw=3, label="Mean"),
        Patch(alpha=0.30, label="IQR (25–75%)"),
        Patch(alpha=0.15, label="5–95% range"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


def write_with_cluster_label_row(writer, sheet_name, data_df):
    data_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
    ws = writer.sheets[sheet_name]
    for col_idx, col_name in enumerate(data_df.columns, start=1):
        if "_cluster" in col_name and "_cell" in col_name:
            k = int(col_name.split("_cluster", 1)[1].split("_cell", 1)[0])
            ws.cell(row=2, column=col_idx, value=f"cluster_{k}")
        else:
            ws.cell(row=2, column=col_idx, value="")


def export_mdz_excel(
    out_xlsx,
    t,
    df,
    CliIC_mat,
    C_IC_bulk,
    elim_rate_mat,
    M_cum_mat,
    kept_idx,
    cluster_id,
    h2_vmax_df=None
):
    venous_df = pd.DataFrame({"Time_min": t, "Cvpl_mg_per_L": df["Cvpl_mg_per_L"].to_numpy()})
    vascular_df = pd.DataFrame({"Time_min": t, "CliV_mg_per_L": df["CliV_mg_per_L"].to_numpy()})
    if_df = pd.DataFrame({"Time_min": t, "CliIF_mg_per_L": df["CliIF_mg_per_L"].to_numpy()})
    ic_bulk_df = pd.DataFrame({"Time_min": t, "C_IC_bulk_mg_per_L": np.asarray(C_IC_bulk)})

    ic_df = pd.DataFrame({"Time_min": t})
    elim_df = pd.DataFrame({"Time_min": t})
    cum_df = pd.DataFrame({"Time_min": t})

    for i in kept_idx:
        k = int(cluster_id[i])
        ic_df[f"IC_cluster{k}_cell{i}"] = CliIC_mat[i, :]
        elim_df[f"ElimRate_cluster{k}_cell{i}"] = elim_rate_mat[i, :]
        cum_df[f"CumMet_cluster{k}_cell{i}"] = M_cum_mat[i, :]

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        venous_df.to_excel(writer, sheet_name="EQ_S1_venous_plasma", index=False)
        vascular_df.to_excel(writer, sheet_name="EQ_S4_liver_vascular", index=False)
        if_df.to_excel(writer, sheet_name="EQ_S5_liver_interstitial", index=False)

        write_with_cluster_label_row(writer, "EQ_S6_liver_intracellular", ic_df)
        ic_bulk_df.to_excel(writer, sheet_name="EQ_S7_liver_intracellular_bulk", index=False)

        write_with_cluster_label_row(writer, "EQ_S8_elimination_rate", elim_df)
        write_with_cluster_label_row(writer, "EQ_S9_cumulative_metabolized", cum_df)

        if h2_vmax_df is not None:
            h2_vmax_df.to_excel(writer, sheet_name="h2_Vmax_interaction", index=False)


if __name__ == "__main__":

    N_tot = int(1.13 * 10**11)   # total hepatocytes
    K = 4

    n_per_cluster = 250
    n_het = K * n_per_cluster

    # for h2 sweep (smaller explicit population)
    n_per_cluster_h2vmax = 250
    n_het_vmax_h2 = K * n_per_cluster_h2vmax

    params = DEFAULT_PARAMS.copy()
    params["h1"] = 27416.0
    params["h2"] = 14606.0

    # ---- Option A: shared mu_count for smoothness ----
    MU_COUNT = 100.0

    # cluster means and thetas
    rng = np.random.default_rng(123)
    mus = rng.uniform(0.6, 2.3, size=K)                 # cluster mean multipliers
    thetas = loguniform(1.0, 20.0, size=K, rng=rng)     # log-uniform theta

    print("Sampled mus:   ", mus)
    print("Sampled thetas:", thetas)
    print("Option A mu_count:", MU_COUNT)

    # explicit cluster labels (balanced)
    cluster_id = sample_clusters(n_het, n_clusters=K, cluster_fracs=[0.25]*4, rng=np.random.default_rng(123))
    cluster_id_sweep = sample_clusters(n_het_vmax_h2, n_clusters=K, cluster_fracs=[0.25]*4, rng=np.random.default_rng(1234))

    # multipliers
    W = sample_clustered_nb_multipliers_optionA(
        cluster_id=cluster_id, mus=mus, thetas=thetas,
        mu_count=MU_COUNT, seed=123
    )
    W_sweep = sample_clustered_nb_multipliers_optionA(
        cluster_id=cluster_id_sweep, mus=mus, thetas=thetas,
        mu_count=MU_COUNT, seed=999
    )

    print("\nCluster summary (multipliers) [baseline]:")
    for k in range(K):
        wk = W[cluster_id == k]
        print(f"  cluster {k}: n={wk.size}, mean={wk.mean():.3f}, std={wk.std():.3f}, min={wk.min():.3f}, max={wk.max():.3f}")

    # ----------------------------
    # Baseline simulation
    # ----------------------------
    sol, df, CliIC_mat, C_IC_bulk, VliIC_i, Vmax_i = simulate_mdz_multicell_with_bulk(
        params=params,
        n_het_cells=n_het,
        N_total_cells=N_tot,
        Vmax_multipliers=W,
        debug=True
    )

    t = df["Time_min"].to_numpy()

    # elimination + cumulative metabolized per cell
    Km   = params["Km"]
    Rli2 = params["Rli2"]
    fup  = params["fup"]
    Ucon = Rli2 / max(fup, 1e-12)

    elim_rate_mat = (Vmax_i[:, None] * CliIC_mat) / (Km * Ucon + CliIC_mat)
    M_cum_mat = cumulative_trapezoid(elim_rate_mat, t, axis=1, initial=0.0)

    # plots
    plot_basic_system_traces(t, df)

    idx_by_cluster = sample_indices_per_cluster(cluster_id, n_show_per_cluster=30, seed=1)
    plot_per_cell_traces(
        t, elim_rate_mat, idx_by_cluster,
        title="Per-cell drug elimination rate (subset, cluster-colored)",
        ylabel="Elimination rate per cell (mg/min)"
    )
    plot_per_cell_traces(
        t, M_cum_mat, idx_by_cluster,
        title="Per-cell cumulative metabolized amount (subset, cluster-colored)",
        ylabel="Cumulative metabolized (mg)"
    )

    h2_base   = float(DEFAULT_PARAMS["h2"])
    vmax_base = float(params["Vmax"])
    h2_scales = np.logspace(-7, -1, 50)

    ratios = []
    metrics = []
    h2_vals = []

    for s in h2_scales:
        p = DEFAULT_PARAMS.copy()
        p.update(params)

        h2_val = h2_base * float(s)
        p["h2"] = h2_val

        ratio = h2_val / vmax_base
        ratios.append(ratio)
        h2_vals.append(h2_val)

        sol_s, df_s, CliIC_mat_s, C_IC_bulk_s, VliIC_i_s, Vmax_i_s = simulate_mdz_multicell_with_bulk(
            params=p,
            n_het_cells=n_het_vmax_h2,
            N_total_cells=N_tot,
            Vmax_multipliers=W_sweep,
            debug=False
        )

        var_t = np.var(CliIC_mat_s, axis=0)
        metrics.append(float(np.mean(var_t)))

    h2_vmax_df = pd.DataFrame({
        "h2_multiplier": h2_scales,
        "h2": h2_vals,
        "Vmax": np.full_like(h2_scales, vmax_base, dtype=float),
        "h2/Vmax": ratios,
        "metric_timeavg_var": metrics
    })

    plt.figure()
    plt.semilogx(ratios, metrics, marker="o")
    plt.xlabel("h2 / Vmax")
    plt.ylabel("Time-avg variance (across cells)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Excel export
    # ----------------------------
    OUT_XLSX = "mdz_liver_outputs.xlsx"
    N_KEEP_PER_CLUSTER = 250
    SEED_SAVE = 123

    kept_idx = pick_indices_balanced_by_cluster(cluster_id, N_KEEP_PER_CLUSTER, seed=SEED_SAVE)
    print(f"\nSaving {kept_idx.size} cells total ({N_KEEP_PER_CLUSTER} per cluster max, K={K}).")
    print(f"Excel output: {OUT_XLSX}")

    export_mdz_excel(
        out_xlsx=OUT_XLSX,
        t=t,
        df=df,
        CliIC_mat=CliIC_mat,
        C_IC_bulk=C_IC_bulk,
        elim_rate_mat=elim_rate_mat,
        M_cum_mat=M_cum_mat,
        kept_idx=kept_idx,
        cluster_id=cluster_id,
        h2_vmax_df=h2_vmax_df
    )

    print("Done.")

    # per-cluster band plot (mean + IQR + 5–95%)
    plot_cluster_timebands(
        t_minutes=t,
        Y_mat=elim_rate_mat,
        cluster_id=cluster_id,
        K=K,
        title="Elimination rate per cluster: mean + IQR + 5–95%",
        y_label="Elim rate (mg/min)",
        x_label="Time (hr)"
    )