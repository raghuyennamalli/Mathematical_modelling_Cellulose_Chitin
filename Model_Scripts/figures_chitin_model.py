"""
 FIGURES - chitin degradation model
 """

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chitin_model import Parameters, Model

T_END_H  = 12.0
N_POINTS = 300
THERMAL  = True

COL = dict(chitin="#4c4c4c", glcnac="#1f77b4", dimer="#d62728",
           oligo="#2ca02c", nolpmo="#1f77b4", lpmo="#d62728", dead="#7f7f7f")


def run(use_lpmo, t_end=T_END_H, **kw):
    m = Model(Parameters(**kw))
    return m.analyse(m.simulate(use_lpmo=use_lpmo, thermal_decay=THERMAL,
                                t_end_h=t_end, n_points=N_POINTS))


def _time_to(d, pct):
    c = d["conversion_pct"]; i = int(np.argmax(c >= pct))
    return d["t"][i] if c[i] >= pct else float("nan")


def figure_conversion(outfile="chitin_fig_conversion.png"):
    base, lp = run(False), run(True)
    t = base["t"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(t, base["conversion_pct"], color=COL["nolpmo"], lw=2.6,
            label="Chitinase + β-N-acetylglucosaminidase  (no LPMO)")
    ax.plot(t, lp["conversion_pct"], color=COL["lpmo"], lw=2.6,
            label="Chitinase + β-N-acetylglucosaminidase + LPMO")
    ax.fill_between(t, base["conversion_pct"], lp["conversion_pct"],
                    where=(lp["conversion_pct"] >= base["conversion_pct"]),
                    color=COL["lpmo"], alpha=0.12, label="LPMO contribution")
    ratio = lp["conversion_pct"][-1] / base["conversion_pct"][-1]
    tb, tl = _time_to(base, 30.0), _time_to(lp, 30.0)

    ax.set_xlim(0, T_END_H)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Chitin solubilized (%)")
    # ax.set_title("Chitin solubilization with and without LPMO", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(outfile, dpi=170); plt.close(fig)
    print(f"[FIG 1] {outfile}   ratio {ratio:.3f}x  ({T_END_H:.0f} h)")


def figure_species(outfile="chitin_fig_species.png"):
    base, lp = run(False), run(True)
    ymax_L = 1.08*max(base["solid_GlcNAc"].max(), lp["solid_GlcNAc"].max())
    ymax_R = 1.20*max(base["chitobiose"].max(), lp["chitobiose"].max(),
                      base["GlcNAc"].max(),     lp["GlcNAc"].max())
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.9))
    for axis, d, title in ((ax[0], base, "A"),
                           (ax[1], lp,   "B")):
        t = d["t"]
        # LEFT y-axis: solid chitin
        l1, = axis.plot(t, d["solid_GlcNAc"], color=COL["chitin"], lw=2.6,
                        label="chitin (solid, GlcNAc equiv.)")
        axis.set_xlim(0, T_END_H); axis.set_ylim(0, ymax_L)
        axis.set_xlabel("Time (h)")
        axis.set_ylabel(" Chitin (mmol L$^{-1}$)")
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.25, lw=0.5)
        # RIGHT y-axis: soluble dimer and monomer on their own scale
        ax2 = axis.twinx()
        l2, = ax2.plot(t, d["chitobiose"], color=COL["dimer"], lw=2.4, ls="-",
                       label="(GlcNAc)$_2$  (chitobiose)")
        l3, = ax2.plot(t, d["GlcNAc"], color=COL["glcnac"], lw=2.4, ls="--",
                       label="GlcNAc (monomer)")
        ax2.set_ylim(0, ymax_R)
        ax2.set_ylabel("Chitobiose/GlcNAC (mmol L$^{-1}$)", color="#555")
        ax2.tick_params(axis="y", labelcolor="#555")
        axis.legend(handles=[l1, l2, l3], fontsize=8.5,
                    loc="center right", framealpha=0.95)
    # fig.suptitle("Reaction species during chitin hydrolysis", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=170); plt.close(fig)
    print(f"[FIG 2] {outfile}")
    for tag, d in (("no LPMO", base), ("+LPMO", lp)):
        print(f"         {tag:8s} GlcNAc {d['GlcNAc'][-1]:5.2f} mM | "
              f"(GlcNAc)2 peak {d['chitobiose'].max():5.2f} mM | "
              f"chito3-6 peak {d['chito_oligomers'].max()*1e3:.3f} uM")


if __name__ == "__main__":
    figure_conversion()
    figure_species()
