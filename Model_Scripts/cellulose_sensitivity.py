"""
Cellulose sensitivity analysis
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import fields
from cellulose_model import Parameters, Model

T_END, NPTS, THERMAL, DELTA = 12.0, 60, True, 0.20   

TAG = {
    "dp_initial":"E", "dp_max":"D", "loading_g_per_L":"E", "surface_area_m2_g":"E",
    "density_g_per_L":"E", "mw_glucose_g_mmol":"E",
    "kcat_EG_per_h":"E", "kcat_CBH_per_h":"E",
    "KM_EG_cellulose":"F", "KM_CBH_cellulose":"F",
    "KM_EG_soluble":"E", "KM_CBH_soluble":"E",
    "Ki_EG_glucose":"F", "Ki_EG_cellobiose":"F", "Ki_CBH_glucose":"E", "Ki_CBH_cellobiose":"F",
    "k_ads_L_mmol_h":"E", "k_des_EG_per_h":"E", "k_des_CBH_per_h":"E",
    "enzyme_footprint":"E", "max_site_density":"E",
    "half_life_EG_h":"F", "half_life_CBH_h":"F", "half_life_BG_h":"A",
    "EG_total_mmol_L":"E", "CBH_total_mmol_L":"E", "BG_total_mmol_L":"A",
    "kcat_BG_per_h":"E", "KM_BG_mmol_L":"E", "kcat_BGox_per_h":"A", "KM_BGox_mmol_L":"A",
    "cellulase_efficiency":"D",
    "LPMO_total_uM":"E", "kcat_LPMO_per_h":"E", "KM_H2O2_uM":"E",
    "S_half_Avicel_g_L":"A", "futile_efficiency_ratio":"A",
    "lpmo_protected_fraction":"A", "p_inactivation_per_futile":"E",
    "H2O2_supply_uM_h":"A", "ascorbate_init_uM":"E", "H2O2_init_uM":"E",
    "k_priming_per_uM_h":"A", "k_ascorbate_autox_per_h":"A",
    "lpmo_C1_fraction":"A", "cbh_activity_on_C1ox":"E",
   
}
NAME = {
    "loading_g_per_L":"$S_0$ loading", "surface_area_m2_g":"surface area $a_s$",
    "mw_glucose_g_mmol":"$M_1$ glucose", "kcat_EG_per_h":"$k_{cat,EG}$",
    "kcat_CBH_per_h":"$k_{cat,CBH}$", "KM_EG_cellulose":"$K^{s}_{M,EG}$",
    "KM_CBH_cellulose":"$K^{s}_{M,CBH}$", "KM_EG_soluble":"$K^{aq}_{M,EG}$",
    "KM_CBH_soluble":"$K^{aq}_{M,CBH}$", "Ki_EG_glucose":"$K_{i,EG}$ glc",
    "Ki_EG_cellobiose":"$K_{i,EG}$ cb", "Ki_CBH_glucose":"$K_{i,CBH}$ glc",
    "Ki_CBH_cellobiose":"$K_{i,CBH}$ cb", "k_ads_L_mmol_h":"$k_{ads}$",
    "k_des_EG_per_h":"$k^{des}_{EG}$", "k_des_CBH_per_h":"$k^{des}_{CBH}$",
    "enzyme_footprint":"footprint $f$", "max_site_density":"$s_{max}$",
    "half_life_EG_h":"$t_{1/2,EG}$", "half_life_CBH_h":"$t_{1/2,CBH}$",
    "EG_total_mmol_L":"[EG]", "CBH_total_mmol_L":"[CBH]", "BG_total_mmol_L":"[BG]",
    "kcat_BG_per_h":"$k_{cat,BG}$", "KM_BG_mmol_L":"$K_{M,BG}$",
    "LPMO_total_uM":"[LPMO]", "kcat_LPMO_per_h":"$k_{cat,LPMO}$",
    "KM_H2O2_uM":"$K_{M,H_2O_2}$", "S_half_Avicel_g_L":"$[S]_{0.5}$",
    "p_inactivation_per_futile":"$p_{inact}$", "H2O2_supply_uM_h":"$q_{H_2O_2}$",
    "lpmo_protected_fraction":r"protected $\pi$", "cbh_activity_on_C1ox":r"$\varphi$",
    "dp_initial":"$DP_0$",
}
BOUNDED01 = {"lpmo_protected_fraction", "lpmo_C1_fraction", "cbh_activity_on_C1ox",
             "futile_efficiency_ratio", "p_inactivation_per_futile"}
STRINGS = {"distribution"}
COL = {"E":"#1f77b4", "F":"#ff7f0e", "A":"#9467bd", "D":"#7f7f7f"}
LEG = {"E":"measured [E]", "F":"fitted by Levine [F]", "A":"assumed [A]", "D":"derived [D]"}


def outputs(**kw):
    m0 = Model(Parameters(**kw))
    d0 = m0.analyse(m0.simulate(use_lpmo=False, thermal_decay=THERMAL, t_end_h=T_END, n_points=NPTS))
    m1 = Model(Parameters(**kw))
    d1 = m1.analyse(m1.simulate(use_lpmo=True, thermal_decay=THERMAL, t_end_h=T_END, n_points=NPTS))
    return d0["conversion_pct"][-1], d1["conversion_pct"][-1]


def run(delta=DELTA, outfile="cellulose_fig_sensitivity.png"):
    p0 = Parameters()
    base0, base1 = outputs()
    print(f"base conversion @12 h:  no-LPMO {base0:.3f} %   +LPMO {base1:.3f} %\n")
    rows, skipped = [], []
    for f in fields(Parameters):
        n = f.name; v0 = getattr(p0, n)
        if n in STRINGS or isinstance(v0, bool) or not isinstance(v0, (int, float)):
            skipped.append(n); continue
        if v0 == 0 or n == "dp_max":
            skipped.append(n); continue
        hi, lo = v0*(1+delta), v0*(1-delta)
        if n in BOUNDED01 and hi > 1.0:
            hi = 1.0
        try:
            c0h, c1h = outputs(**{n: hi}); c0l, c1l = outputs(**{n: lo})
        except Exception:
            skipped.append(n); continue
        dP = (hi-lo)/v0
        S0 = ((c0h-c0l)/base0)/dP; S1 = ((c1h-c1l)/base1)/dP
        rows.append(dict(key=n, label=NAME.get(n, n), tag=TAG.get(n, "?"), S0=S0, S1=S1))
        print(f"  {NAME.get(n,n):22s} [{TAG.get(n,'?')}]  S0={S0:+8.4f}  S1={S1:+8.4f}")
    print(f"\ncovered {len(rows)}; skipped {len(skipped)}")

    rows.sort(key=lambda r: max(abs(r["S0"]), abs(r["S1"])))
    labels = [r["label"] + f"  [{r['tag']}]" for r in rows]
    S0 = np.array([r["S0"] for r in rows]); S1 = np.array([r["S1"] for r in rows])
    cols = [COL.get(r["tag"], "#333") for r in rows]; y = np.arange(len(rows))
    lim = 1.05*max(np.abs(np.concatenate([S0, S1])).max(), 1e-6)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, max(9, 0.32*len(rows))))
    for axis, S, title in ((ax[0], S0, "A"), (ax[1], S1, "B")):
        axis.barh(y, S, color=cols, alpha=0.85)
        axis.set_yticks(y); axis.set_yticklabels(labels, fontsize=8)
        axis.axvline(0, color="k", lw=0.8); axis.set_xlim(-lim, lim)
        axis.set_xlabel("Normalised sensitivity  S = (ΔY/Y)/(ΔP/P)")
    #    axis.set_title(title + "\noutput: cellulose solubilized at 12 h", fontsize=11)
        axis.grid(alpha=0.25, lw=0.5, axis="x")
    from matplotlib.patches import Patch
    seen = [t for t in ("E", "F", "A", "D") if t in {r["tag"] for r in rows}]
    ax[1].legend(handles=[Patch(color=COL[t], label=LEG[t]) for t in seen],
                 fontsize=8.5, loc="lower right", framealpha=0.95)
    #fig.suptitle(f"Cellulose model - local sensitivity ({len(rows)} parameters"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(outfile, dpi=170); plt.close(fig)
    print(f"\n[FIG] {outfile}")
    tot = {t: 0.0 for t in COL}
    for r in rows:
        tot[r["tag"]] += abs(r["S1"])
    grand = sum(tot.values()) or 1.0
    print("\nSHARE OF TOTAL |S| (+LPMO) BY PROVENANCE:")
    for t in ("E", "F", "A", "D"):
        print(f"   {LEG[t]:24s} {100*tot[t]/grand:5.1f} %")


if __name__ == "__main__":
    run()