"""
Chitin sensitivity analysis
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import fields
from chitin_model import Parameters, Model

T_END, NPTS, THERMAL, DELTA = 12.0, 60, True, 0.20   

TAG = {
    "dp_initial":"A", "dp_max":"D", "loading_g_per_L":"E", "surface_area_m2_g":"E",
    "density_g_per_L":"E", "mw_GlcNAc_g_mmol":"E",
    "kcat_exo_per_h":"E", "kcat_endo_per_h":"M", "KM_exo_solid":"M",
    "KM_endo_solid":"M", "KM_exo_soluble":"E", "KM_endo_soluble":"A",
    "Ki_endo_GlcNAc":"M", "Ki_endo_GlcNAc2":"M", "Ki_exo_GlcNAc":"M", "Ki_exo_GlcNAc2":"M",
    "k_ads_L_mmol_h":"M", "k_des_endo_per_h":"M", "k_des_exo_per_h":"M",
    "enzyme_footprint":"M", "max_site_density":"M",
    "half_life_endo_h":"E", "half_life_exo_h":"E", "half_life_GH20_h":"P",
    "endo_total_mmol_L":"M", "exo_total_mmol_L":"M", "GH20_total_mmol_L":"A",
    "kcat_GH20_per_h":"A", "KM_GH20_mmol_L":"A", "kcat_GH20ox_per_h":"E", "KM_GH20ox_mmol_L":"D",
    "chitinase_efficiency":"D",
    "LPMO_total_uM":"E", "kcat_LPMO_per_h":"E", "KM_H2O2_uM":"E",
    "S_half_chitin_g_L":"M", "futile_efficiency_ratio":"A",
    "lpmo_protected_fraction":"A", "p_inactivation_per_futile":"E",
    "H2O2_supply_uM_h":"A", "ascorbate_init_uM":"E", "H2O2_init_uM":"E",
    "k_priming_per_uM_h":"A", "k_reoxidation_per_h":"A", "k_ascorbate_autox_per_h":"A",
    "lpmo_C1_fraction":"E", "exo_activity_on_C1ox":"M",
    "bimodal_radius_ratio":"D", "bimodal_small_fraction":"D",
}
NAME = {
    "loading_g_per_L":"$S_0$ (chitin loading)", "surface_area_m2_g":"surface area $a_s$",
    "density_g_per_L":r"$\rho$ (chitin)", "mw_GlcNAc_g_mmol":"$M_1$ (GlcNAc mass)",
    "kcat_exo_per_h":"$k_{cat,exo}$ (ChiA)", "kcat_endo_per_h":"$k_{cat,endo}$ (ChiC)",
    "KM_exo_solid":"$K^{sol}_{M,exo}$", "KM_endo_solid":"$K^{sol}_{M,endo}$",
    "KM_exo_soluble":"$K^{aq}_{M,exo}$", "KM_endo_soluble":"$K^{aq}_{M,endo}$",
    "Ki_endo_GlcNAc":"$K_{i,endo}$ GlcNAc", "Ki_endo_GlcNAc2":"$K_{i,endo}$ (GlcNAc)$_2$",
    "Ki_exo_GlcNAc":"$K_{i,exo}$ GlcNAc", "Ki_exo_GlcNAc2":"$K_{i,exo}$ (GlcNAc)$_2$",
    "k_ads_L_mmol_h":"$k_{ads}$", "k_des_endo_per_h":"$k^{des}_{endo}$",
    "k_des_exo_per_h":"$k^{des}_{exo}$", "enzyme_footprint":"footprint $f$",
    "max_site_density":"$s_{max}$", "half_life_endo_h":"$t_{1/2,endo}$",
    "half_life_exo_h":"$t_{1/2,exo}$", "half_life_GH20_h":"$t_{1/2,GH20}$",
    "endo_total_mmol_L":"[endo]", "exo_total_mmol_L":"[exo]", "GH20_total_mmol_L":"[GH20]",
    "kcat_GH20_per_h":"$k_{cat,GH20}$", "KM_GH20_mmol_L":"$K_{M,GH20}$",
    "LPMO_total_uM":"[LPMO]", "kcat_LPMO_per_h":"$k_{cat,LPMO}$",
    "KM_H2O2_uM":"$K_{M,H_2O_2}$", "S_half_chitin_g_L":"$[S]_{0.5}$",
    "futile_efficiency_ratio":r"futile ratio $\gamma$",
    "p_inactivation_per_futile":"$p_{inact}$", "H2O2_supply_uM_h":"$q_{H_2O_2}$",
    "ascorbate_init_uM":"[AscA]$_0$", "k_priming_per_uM_h":"$k_{prim}$",
    "lpmo_protected_fraction":r"protected frac. $\pi$",
    "lpmo_C1_fraction":"C1 fraction", "exo_activity_on_C1ox":r"$\varphi$ (exo on ox.)",
    "dp_initial":"$DP_0$",
}
BOUNDED01 = {"lpmo_protected_fraction", "lpmo_C1_fraction", "exo_activity_on_C1ox",
             "futile_efficiency_ratio", "p_inactivation_per_futile"}
COL = {"E":"#1f77b4", "M":"#d62728", "A":"#9467bd", "D":"#7f7f7f"}
LEG = {"E":"measured [E]", "M":"Missing [M]",
       "A":"assumed [A]", "D":"derived [D]"}


def outputs(**kw):
    m0 = Model(Parameters(**kw))
    d0 = m0.analyse(m0.simulate(use_lpmo=False, thermal_decay=THERMAL, t_end_h=T_END, n_points=NPTS))
    m1 = Model(Parameters(**kw))
    d1 = m1.analyse(m1.simulate(use_lpmo=True, thermal_decay=THERMAL, t_end_h=T_END, n_points=NPTS))
    return d0["conversion_pct"][-1], d1["conversion_pct"][-1]


def run(delta=DELTA, outfile="chitin_fig_sen.png"):
    p0 = Parameters()
    base0, base1 = outputs()
    print(f"base conversion @12 h:  no-LPMO {base0:.3f} %   +LPMO {base1:.3f} %\n")

    rows, skipped = [], []
    for f in fields(Parameters):
        n = f.name; v0 = getattr(p0, n)
        if isinstance(v0, bool) or not isinstance(v0, (int, float)):
            skipped.append((n, "non-numeric")); continue
        if v0 == 0:
            skipped.append((n, "nominal 0")); continue
        if n == "dp_max":
            skipped.append((n, "integer truncation setting")); continue
        hi, lo, onesided = v0*(1+delta), v0*(1-delta), False
        if n in BOUNDED01 and hi > 1.0:
            hi, onesided = 1.0, True
        try:
            c0h, c1h = outputs(**{n: hi}); c0l, c1l = outputs(**{n: lo})
        except Exception as e:
            skipped.append((n, f"failed {type(e).__name__}")); continue
        dP = (hi-lo)/v0
        S0 = ((c0h-c0l)/base0)/dP
        S1 = ((c1h-c1l)/base1)/dP
        rows.append(dict(key=n, label=NAME.get(n, n), tag=TAG.get(n, "?"), S0=S0, S1=S1))
        print(f"  {NAME.get(n,n):24s} [{TAG.get(n,'?')}]  S(no-LPMO)={S0:+8.4f}  S(+LPMO)={S1:+8.4f}")

    print(f"\ncovered {len(rows)} parameters; skipped {len(skipped)}")
    for n, why in skipped:
        print(f"   - {n}: {why}")

    rows.sort(key=lambda r: max(abs(r["S0"]), abs(r["S1"])))
    labels = [r["label"] + f"  [{r['tag']}]" for r in rows]
    S0 = np.array([r["S0"] for r in rows]); S1 = np.array([r["S1"] for r in rows])
    cols = [COL.get(r["tag"], "#333") for r in rows]
    y = np.arange(len(rows))
    lim = 1.05*max(np.abs(np.concatenate([S0, S1])).max(), 1e-6)

    fig, ax = plt.subplots(1, 2, figsize=(14.5, max(9, 0.32*len(rows))))
    for axis, S, title in ((ax[0], S0, "A"),
                           (ax[1], S1, "B")):
        axis.barh(y, S, color=cols, alpha=0.85)
        axis.set_yticks(y); axis.set_yticklabels(labels, fontsize=8)
        axis.axvline(0, color="k", lw=0.8); axis.set_xlim(-lim, lim)
        axis.set_xlabel("Normalised sensitivity  S = (ΔY/Y)/(ΔP/P)")
        axis.set_title(title + "\noutput: chitin solubilized at 12 h", fontsize=11)
        axis.grid(alpha=0.25, lw=0.5, axis="x")
    from matplotlib.patches import Patch
    seen = [t for t in ("E", "M", "A", "D") if t in {r["tag"] for r in rows}]
    ax[1].legend(handles=[Patch(color=COL[t], label=LEG[t]) for t in seen],
                 fontsize=8.5, loc="lower right", framealpha=0.95)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, dpi=170); plt.close(fig)
    print(f"\n[FIG] {outfile}")

    # provenance summary: how much total sensitivity sits on measured vs not
    tot = {t: 0.0 for t in list(COL) + ["?"]}
    for r in rows:
        tot.setdefault(r["tag"], 0.0)
        tot[r["tag"]] += abs(r["S1"])
    grand = sum(tot.values()) or 1.0
    print("\nSHARE OF TOTAL |S| (+LPMO) BY PROVENANCE:")
    for t in ("E", "M", "A", "D"):
        label = LEG.get(t, t)
        print(f"   {label:28s} {100*tot.get(t, 0.0)/grand:5.1f} %")
    if tot.get("?", 0.0) > 0:
        print(f"   {'untagged':28s} {100*tot['?']/grand:5.1f} %")
    return rows


if __name__ == "__main__":
    run()