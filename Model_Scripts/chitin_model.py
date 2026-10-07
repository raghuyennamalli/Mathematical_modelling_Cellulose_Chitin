"""
A MECHANISTIC MODEL OF ENZYMATIC CHITIN DEGRADATION
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import poisson

#PARAMETERS

@dataclass
class Parameters:
    #substrate: crystalline chitin 
    dp_initial:        float = 300.0    
    dp_max:            int   = 350     
    loading_g_per_L:   float = 10.0                                              
    surface_area_m2_g: float = 47.6      
    density_g_per_L:   float = 1425.0                                             
    mw_GlcNAc_g_mmol:   float = 0.20319  
    kcat_exo_per_h:  float = 2.6 * 3600.0                                          
    kcat_endo_per_h: float = 2.6 * 3600.0   
    KM_exo_solid:    float = 0.64                                                                                            
    KM_endo_solid:   float = 0.0067       
    KM_exo_soluble:  float = 0.009         
    KM_endo_soluble: float = 0.004         
    Ki_endo_GlcNAc:  float = 16.9   
    Ki_endo_GlcNAc2: float = 0.01   
    Ki_exo_GlcNAc:   float = 31.0   
    Ki_exo_GlcNAc2:  float = 0.093   
    k_ads_L_mmol_h:  float = 8640.0  
    k_des_endo_per_h:float = 19.3    
    k_des_exo_per_h: float = 164.0   
    enzyme_footprint:float = 156.0   
    max_site_density:float = 6.64e-5 
    half_life_endo_h: float = 240.0  
    half_life_exo_h:  float = 240.0  
    half_life_GH20_h: float = 240.0  
    # enzyme loadings
    endo_total_mmol_L: float = 1.6e-3 
    exo_total_mmol_L:  float = 1.6e-3 
    GH20_total_mmol_L: float = 0.16e-3
    kcat_GH20_per_h:  float = 1.41 * 60.0   
    KM_GH20_mmol_L:   float = 0.213         
    kcat_GH20ox_per_h:float = 0.0          
    KM_GH20ox_mmol_L: float = 0.5        
    chitinase_efficiency: float = 1.0                                            
    LPMO_total_uM:      float = 1.0     
    kcat_LPMO_per_h:    float = 6.7 * 3600.0                                      
    KM_H2O2_uM:         float = 3.0    
    S_half_chitin_g_L:  float = 3.0                                              
    futile_efficiency_ratio: float = 0.10                                                
    lpmo_protected_fraction: float = 0.98                                               
    p_inactivation_per_futile: float = 0.001 
    H2O2_supply_uM_h:   float = 83.4     
    ascorbate_init_uM:  float = 100.0   
    H2O2_init_uM:       float = 0.0      
    k_priming_per_uM_h: float = 3.2*3600.0/2.0  
    k_reoxidation_per_h:float = 0.0      
    k_ascorbate_autox_per_h: float = 0.02
    lpmo_C1_fraction:   float = 1.0      
    exo_activity_on_C1ox: float = 0.68  
#solid chains  C'_7..C'_DPmax           (mmol chains / dm^2)
#soluble   C1..C6  (GlcNAc..(GlcNAc)6, mmol/L)
#glcnac_ox  chitobionic_acid  oxidised_oligomers   (mmol/L, GlcNAc-equiv)
#ends_ox_C1    (mmol/L; C1-oxidised solid ends)
#Cu_II Cu_I Cu_dead H2O2 ascorbate      (uM)
#endo_ads(endo)  exo_ads(exo)       (mmol/dm^2)
#radius    (dm)
class Model:
    def __init__(self, p: Parameters):
        self.p = p
        self.dp = np.arange(7, p.dp_max + 1)  
        self.n_solid = self.dp.size
        n = self.n_solid
        self.I_solid   = slice(0, n)
        self.I_soluble = slice(n, n + 6)
        self.i_GlcNAc_ox   = n + 6  
        self.i_chitobionic = n + 7  
        self.i_oligo_ox    = n + 8   
        self.i_ends_ox     = n + 9   
        self.i_CuII        = n + 10
        self.i_CuI         = n + 11
        self.i_Cudead      = n + 12  
        self.i_H2O2        = n + 13
        self.i_ascorbate   = n + 14
        self.i_endo_ads    = n + 15  
        self.i_exo_ads     = n + 16  
        self.i_radius      = n + 17
        self.n_state       = n + 18

        # total GlcNAc in the system (mmol/L) 
        self.total_GlcNAc = p.loading_g_per_L / p.mw_GlcNAc_g_mmol

        # initial solid chain-length distribution ~ Poisson(DP0), normalised
        self.chain_pmf = poisson.pmf(self.dp, p.dp_initial)
        self.chain_pmf /= self.chain_pmf.sum()
        self._setup_geometry()

    #particle geometry 
    def _setup_geometry(self):
        p = self.p
        sa_dm2_g = p.surface_area_m2_g * 100.0                 # m^2/g -> dm^2/g
        self.radius0 = 3.0 / (sa_dm2_g * p.density_g_per_L)    # Eq.2 solved for R
        area_per_L0  = sa_dm2_g * p.loading_g_per_L            # dm^2/L at t=0
        self.n_particles = area_per_L0 / (4.0*np.pi*self.radius0**2)   # Eq.A1 -> N (const)

    def area_per_volume(self, radius):
        return 4.0*np.pi*radius**2*self.n_particles          

    def initial_state(self):
        p = self.p
        y = np.zeros(self.n_state)
        chains_per_area = p.max_site_density / p.dp_initial   
        y[self.I_solid]  = chains_per_area * self.chain_pmf
        y[self.i_CuII]      = p.LPMO_total_uM                 
        y[self.i_ascorbate] = p.ascorbate_init_uM
        y[self.i_H2O2]      = p.H2O2_init_uM
        y[self.i_radius]    = self.radius0
        return y

    def _inhibition_terms(self, soluble):
        p = self.p
        GlcNAc, dimer = soluble[0], soluble[1]     
        inhib_endo_sol = (np.sum((np.arange(3, 7) - 1.0)*soluble[2:6]) / p.KM_endo_soluble
                        + GlcNAc/p.Ki_endo_GlcNAc + dimer/p.Ki_endo_GlcNAc2)
        inhib_exo_sol = (np.sum(soluble[2:6]) / p.KM_exo_soluble
                         + GlcNAc/p.Ki_exo_GlcNAc + dimer/p.Ki_exo_GlcNAc2)
        return inhib_endo_sol, inhib_exo_sol

    #  Surface site balance  (Levine Eqs A5-A7)                          
    #  Substituting A6/A7 into A5 makes it LINEAR in theta -> closed form 

    def site_balance(self, endo_ads, exo_ads, solid, soluble):
        p = self.p
        occ_endo  = p.enzyme_footprint * endo_ads  / p.max_site_density
        occ_exo = p.enzyme_footprint * exo_ads / p.max_site_density
        bonds_solid  = np.sum((self.dp - 1.0) * solid)     
        chains_solid = np.sum(solid)                       
        inhib_endo_sol, inhib_exo_sol = self._inhibition_terms(soluble)

        a = (1.0
             - occ_endo *(1.0 + occ_endo *bonds_solid /p.KM_endo_solid  + inhib_endo_sol)
             - occ_exo*(1.0 + occ_exo*chains_solid/p.KM_exo_solid + inhib_exo_sol))
        b = (occ_endo *bonds_solid /p.KM_endo_solid
             + occ_exo*chains_solid/p.KM_exo_solid)
        theta = a / (1.0 + b)
        theta = min(max(theta, 0.0), 1.0)  
        return theta, min(max(theta+occ_endo,0.0),1.0), min(max(theta+occ_exo,0.0),1.0)

    
    def lpmo_cycle(self, CuII, CuI, H2O2, ascorbate, solid_conc_g_L):
        """Productive branch is SATURABLE Michaelis-Menten in H2O2 (kcat and
        Km(H2O2) known separately: 5.6 s^-1 and ~3 uM for SmAA10A).  Substrate
        availability enters through the [S]0.5 term (Kont 2020 Eq.7)."""
        p = self.p
        accessibility = solid_conc_g_L / (solid_conc_g_L + p.S_half_chitin_g_L)
        v_priming = p.k_priming_per_uM_h * CuII * ascorbate
        v_peroxygenase = (p.kcat_LPMO_per_h * CuI * H2O2
                          / (p.KM_H2O2_uM + H2O2)) * accessibility
        free_frac = (1.0 - p.lpmo_protected_fraction)
        v_red_perox = (p.futile_efficiency_ratio * p.kcat_LPMO_per_h * CuI * H2O2
                       / (p.KM_H2O2_uM + H2O2)) * free_frac
        v_inactivation = p.p_inactivation_per_futile * v_red_perox

        dCuII = -v_priming
        dCuI  = +v_priming - v_inactivation
        dCudead = +v_inactivation
        dH2O2 = p.H2O2_supply_uM_h - v_peroxygenase - v_red_perox
        dAsc  = -v_priming - v_red_perox - p.k_ascorbate_autox_per_h*ascorbate
        cut_rate_mmol_L_h = v_peroxygenase * 1e-3               
        return cut_rate_mmol_L_h, dCuII, dCuI, dCudead, dH2O2, dAsc, accessibility

    def rhs(self, t, y, use_lpmo, thermal_decay):
        p  = self.p
        dp = self.dp
        solid    = np.maximum(y[self.I_solid],   0.0)
        soluble  = np.maximum(y[self.I_soluble], 0.0)
        chitobionic  = max(y[self.i_chitobionic], 0.0)   
        ends_ox      = max(y[self.i_ends_ox],     0.0)
        CuII = max(y[self.i_CuII], 0.0); CuI = max(y[self.i_CuI], 0.0)
        H2O2 = max(y[self.i_H2O2], 0.0); ascorbate = max(y[self.i_ascorbate], 0.0)
        endo_ads  = max(y[self.i_endo_ads],  0.0)           
        exo_ads = max(y[self.i_exo_ads], 0.0)           
        radius  = max(y[self.i_radius], 1e-12)
        dimer = soluble[1]                          
        area_vol = self.area_per_volume(radius)          

        # thermal deactivation of the hydrolytic enzymes 
        decay = (lambda h: np.exp(-np.log(2)*t/h)) if thermal_decay else (lambda h: 1.0)
        endo_tot  = p.endo_total_mmol_L  * decay(p.half_life_endo_h)
        exo_tot = p.exo_total_mmol_L * decay(p.half_life_exo_h)
        GH20_tot  = p.GH20_total_mmol_L  * decay(p.half_life_GH20_h)

        # site balance and free enzyme 
        theta, theta_endo, theta_exo = self.site_balance(endo_ads, exo_ads, solid, soluble)
        free_sites = p.max_site_density * theta
        inhib_endo_sol, inhib_exo_sol = self._inhibition_terms(soluble)
       
        # full A8/A9 accounting-subtract ALL surface-bound forms
        bracket_endo  = (1.0 + theta_endo *np.sum((dp-1.0)*solid)/p.KM_endo_solid
                       + inhib_endo_sol)
        bracket_exo = (1.0 + theta_exo*np.sum(solid)        /p.KM_exo_solid
                       + inhib_exo_sol)
        endo_free  = max(endo_tot  - area_vol*endo_ads *bracket_endo,  0.0)   
        exo_free = max(exo_tot - area_vol*exo_ads*bracket_exo, 0.0)   

        # adsorption ODEs
        dendo_ads  = p.k_ads_L_mmol_h*endo_free *free_sites - p.k_des_endo_per_h *endo_ads
        dexo_ads = p.k_ads_L_mmol_h*exo_free*free_sites - p.k_des_exo_per_h*exo_ads

        s = p.chitinase_efficiency
        endo_scission     = s*(p.kcat_endo_per_h/p.KM_endo_solid)  *endo_ads *theta_endo  
        exo_removal_nat = s*(p.kcat_exo_per_h/p.KM_exo_solid)*exo_ads*theta_exo  

        # substrate-depletion guard 
        depl = min(max(radius/(1.0e-3*self.radius0), 0.0), 1.0)
        endo_scission     *= depl
        exo_removal_nat *= depl

        # exo-chitinase impediment on C1-oxidised ends (Keller) 
        total_ends_vol = np.sum(solid)*area_vol
        frac_ox = min(ends_ox/total_ends_vol, 1.0) if total_ends_vol > 1e-30 else 0.0
        phi = p.exo_activity_on_C1ox
        exo_factor  = (1.0 - frac_ox) + phi*frac_ox             
        exo_removal = exo_removal_nat * exo_factor
        # fraction of exo turnovers on oxidised ends -> chitobionic acid not (GlcNAc)2
        ox_turnover_share = (phi*frac_ox/exo_factor) if exo_factor > 1e-30 else 0.0

        # LPMO cycle -> oxidative cut rate 
        lpmo_cut_rate = 0.0
        dCuII=dCuI=dCudead=dH2O2=dAsc=0.0
        solid_conc_g_L = p.loading_g_per_L * (radius/self.radius0)**3
        if use_lpmo:
            (lpmo_cut_rate, dCuII, dCuI, dCudead, dH2O2, dAsc) = self.lpmo_cycle(CuII, CuI, H2O2, ascorbate, solid_conc_g_L)
            lpmo_cut_rate *= depl

        total_solid_bonds = np.sum((dp-1.0)*solid) * area_vol
        lpmo_scission = (lpmo_cut_rate/total_solid_bonds) if total_solid_bonds > 1e-30 else 0.0

        # solid chain-length balance 
        cum_longer  = np.concatenate([np.cumsum(solid[::-1])[::-1][1:], [0.0]])  
        solid_plus2 = np.concatenate([solid[2:], [0.0, 0.0]])                   
        endo_term   = endo_scission   * (2.0*cum_longer - (dp-1.0)*solid)   
        lpmo_term = lpmo_scission * (2.0*cum_longer - (dp-1.0)*solid)   
        exo_term  = exo_removal   * (solid_plus2 - solid)              
        dsolid_reaction = endo_term + lpmo_term + exo_term

        # soluble fragments released from the solid 
        longer_than = np.array([np.sum(solid[dp > k]) for k in range(0, 7)])
        native_release   = np.zeros(7)
        oxidised_release = np.zeros(7)
        f_C1 = p.lpmo_C1_fraction
        for k in range(1, 7):
            native_release[k]   = (2.0*endo_scission*longer_than[k]
                                   + (2.0 - f_C1)*lpmo_scission*longer_than[k])
            oxidised_release[k] = f_C1*lpmo_scission*longer_than[k]
        # exo enzyme releases the dimer (GlcNAc)2 per turnover; DP7->C5, DP8->C6
        exo_dimer = exo_removal*np.sum(solid)
        native_release[5] += exo_removal*solid[0]
        native_release[6] += exo_removal*solid[1]

        exo_dimer_vol   = exo_dimer*area_vol
        chitobionic_from_exo = ox_turnover_share*exo_dimer_vol       
        native_dimer_exo= (1.0-ox_turnover_share)*exo_dimer_vol 

        areal_GlcNAc_flux = (np.sum([k*native_release[k]   for k in range(1,7)])
                              + np.sum([k*oxidised_release[k] for k in range(1,7)])
                              + 2.0*exo_dimer)

        # new-chain exposure as the particle recedes 
        dsolid_exposure = self.chain_pmf * (areal_GlcNAc_flux/p.dp_initial)
        dsolid = dsolid_reaction + dsolid_exposure

        #  Soluble-phase balances 
        
        d_soluble = np.zeros(6)
        for k in range(1, 7):
            d_soluble[k-1] += native_release[k]*area_vol
        d_soluble[1] += native_dimer_exo                    
        # soluble-phase exo: C_k -> C_{k-2} + (GlcNAc)2  (adsorbed + free)
        exo_soluble = (exo_free + exo_ads*area_vol)*(p.kcat_exo_per_h/p.KM_exo_soluble)
        for k in (6, 5, 4, 3):
            v = exo_soluble*soluble[k-1]
            d_soluble[k-1] -= v; d_soluble[k-3] += v; d_soluble[1] += v
        # soluble-phase endo random scission of C3..C6 (mass-conserving)
        endo_soluble = (endo_free + endo_ads*area_vol)*(p.kcat_endo_per_h/p.KM_endo_soluble)
        for k in (6, 5, 4, 3):
            per_bond = endo_soluble*soluble[k-1]
            d_soluble[k-1] -= (k-1.0)*per_bond
            for cut in range(1, k):
                d_soluble[cut-1]     += per_bond
                d_soluble[k-cut-1]   += per_bond
        
        v_GH20 = (p.kcat_GH20_per_h*GH20_tot)*dimer/(p.KM_GH20_mmol_L + dimer)
        d_soluble[1] -= v_GH20; d_soluble[0] += 2.0*v_GH20 

        # oxidised product pools
        d_GlcNAc_ox    = oxidised_release[1]*area_vol             
        d_chitobionic = oxidised_release[2]*area_vol + chitobionic_from_exo  
        d_oligo_ox    = np.sum([k*oxidised_release[k] for k in range(3,7)])*area_vol
        
        # C1-oxidised reducing ends on the solid 
        d_ends_ox = 0.0
        if use_lpmo:
            ends_created  = lpmo_cut_rate * p.lpmo_C1_fraction
            ends_released = np.sum(oxidised_release[1:7])*area_vol
            d_ends_ox = ends_created - ends_released - chitobionic_from_exo

        # particle radius (Levine Eq.1) 
        d_radius = -(p.mw_GlcNAc_g_mmol/p.density_g_per_L)*areal_GlcNAc_flux

        # assemble derivative vector 
        dydt = np.zeros(self.n_state)
        dydt[self.I_solid]   = dsolid
        dydt[self.I_soluble] = d_soluble
        dydt[self.i_GlcNAc_ox]    = d_GlcNAc_ox
        dydt[self.i_chitobionic] = d_chitobionic
        dydt[self.i_oligo_ox]    = d_oligo_ox
        dydt[self.i_ends_ox]     = d_ends_ox
        dydt[self.i_CuII]  = dCuII; dydt[self.i_CuI] = dCuI; dydt[self.i_Cudead] = dCudead
        dydt[self.i_H2O2]  = dH2O2; dydt[self.i_ascorbate] = dAsc
        dydt[self.i_endo_ads]  = dendo_ads
        dydt[self.i_exo_ads] = dexo_ads
        dydt[self.i_radius]  = d_radius
        return dydt
                                              
    #  Integrate 
    def _atol_vector(self):
        """Per-state absolute tolerances: the state spans ~10 orders of
        magnitude, so a single scalar atol over-resolves the tiny solid states."""
        atol = np.empty(self.n_state)
        atol[self.I_solid]   = 1e-12
        atol[self.I_soluble] = 1e-8
        atol[self.i_GlcNAc_ox:self.i_ends_ox+1] = 1e-8
        atol[self.i_CuII:self.i_H2O2+1] = 1e-6
        atol[self.i_ascorbate] = 1e-4
        atol[self.i_endo_ads]  = 1e-12
        atol[self.i_exo_ads] = 1e-12
        atol[self.i_radius]  = 1e-14
        return atol

    def simulate(self, use_lpmo=False, thermal_decay=True, t_end_h=12.0,
                 n_points=200, max_step=np.inf):
        # BDF: the coupled adsorption/site-balance system is stiff.
        self._setup_geometry()
        sol = solve_ivp(self.rhs, (0.0, t_end_h), self.initial_state(),
                        args=(use_lpmo, thermal_decay), method="BDF",
                        rtol=1e-6, atol=self._atol_vector(), max_step=max_step,
                        t_eval=np.linspace(0.0, t_end_h, n_points))
        return sol


    #  Post-processing                                                   
    def analyse(self, sol):
        r = sol.y[self.i_radius]
        soluble = sol.y[self.I_soluble]
        GlcNAc_ox   = sol.y[self.i_GlcNAc_ox]
        chitobionic = sol.y[self.i_chitobionic]
        oligo_ox    = sol.y[self.i_oligo_ox]
        # solid GlcNAc from the particle volume (independent of soluble pools)
        solid_GlcNAc = (4.0/3.0)*np.pi*r**3*self.p.density_g_per_L*self.n_particles/self.p.mw_GlcNAc_g_mmol
        native_soluble = np.sum([(k+1)*soluble[k] for k in range(6)], axis=0)
        oxidised_soluble = 1.0*GlcNAc_ox + 2.0*chitobionic + oligo_ox        # GlcNAc-equiv
        total = solid_GlcNAc + native_soluble + oxidised_soluble
        return dict(
            t=sol.t, radius=r,
            solid_GlcNAc=solid_GlcNAc,          
            GlcNAc=soluble[0],                 
            chitobiose=soluble[1],             
            chito3=soluble[2], chito4=soluble[3],
            chito5=soluble[4], chito6=soluble[5],
            chito_oligomers=soluble[2:].sum(0),
            soluble_all=soluble,
            GlcNAc_ox=GlcNAc_ox,                
            chitobionic_acid=chitobionic,      
            oxidised_oligomers=oligo_ox,
            conversion_pct=100.0*(1.0 - solid_GlcNAc/self.total_GlcNAc),
            total_GlcNAc=total,
            mass_error=np.max(np.abs(total - self.total_GlcNAc)),
            CuII=sol.y[self.i_CuII], CuI=sol.y[self.i_CuI], Cu_dead=sol.y[self.i_Cudead],
            H2O2=sol.y[self.i_H2O2], ascorbate=sol.y[self.i_ascorbate],
            ends_ox=sol.y[self.i_ends_ox],
            h2o2_valid_until_h=self._h2o2_validity(sol))

    def _h2o2_validity(self, sol, km_multiple=1.0):
        
        km_h2o2_uM = self.p.KM_H2O2_uM * km_multiple
        h = sol.y[self.i_H2O2]
        idx = np.argmax(h >= km_h2o2_uM)
        return float(sol.t[idx]) if h[idx] >= km_h2o2_uM else None

if __name__ == "__main__":
    P = Parameters()
    print("CHITIN DEGRADATION MODEL - self-check")
    print("=" * 70)
    print(f"monomer M1 (GlcNAc)    = {P.mw_GlcNAc_g_mmol:.5f} g/mmol (203.19 g/mol)")
    print(f"density (alpha-chitin) = {P.density_g_per_L:.0f} g/L")
    print(f"endo-chitinase kcat    = {P.kcat_endo_per_h/3600:.1f} s^-1  (ChiC endo placeholder)")
    print(f"exo-chitinase kcat     = {P.kcat_exo_per_h/3600:.1f} s^-1  (ChiA crystalline, Nakamura 2018)")
    print(f"GH20 GlcNAcase kcat    = {P.kcat_GH20_per_h/3600:.3f} s^-1  (Enterobacter, ref5)")
    print(f"LPMO kcat / Km(H2O2)   = {P.kcat_LPMO_per_h/3600:.1f} s^-1 / {P.KM_H2O2_uM:.0f} uM  (SmAA10A, Kuusk 2018)")
    print(f"chitinase half-life    = {P.half_life_exo_h:.0f} h  (Brurberg 1996)")

    print("\n1) HYDROLYTIC BASELINE  (GH20 off, LPMO off, thermal off)")
    print("   endo + exo chitinase only; checks the solver and mass conservation")
    for sa in (8.0, 47.6):
        m = Model(Parameters(GH20_total_mmol_L=0.0, surface_area_m2_g=sa))
        d = m.analyse(m.simulate(use_lpmo=False, thermal_decay=False,
                                 t_end_h=48.0, n_points=60))
        print(f"   SA={sa:5.1f} m2/g : conversion@48h = {d['conversion_pct'][-1]:6.2f} %"
              f"   mass_err = {d['mass_error']:.1e}")

    print("\n2) FULL CHITIN SYSTEM  (endo + exo + GH20 + LPMO, thermal ON, 12 h)")
    for lpmo in (False, True):
        m = Model(Parameters())
        d = m.analyse(m.simulate(use_lpmo=lpmo, thermal_decay=True,
                                 t_end_h=12.0, n_points=120))
        tag = ""
        if lpmo:
            tag = (f" | GlcNAc-ox {d['GlcNAc_ox'][-1]:.3f} mM"
                   f" | LPMO inact {100*d['Cu_dead'][-1]/P.LPMO_total_uM:.1f}%")
        print(f"   LPMO={str(lpmo):5s} conv@12h={d['conversion_pct'][-1]:6.2f}%"
              f" | GlcNAc {d['GlcNAc'][-1]:5.2f} mM"
              f" | (GlcNAc)2 peak {d['chitobiose'].max()*1e3:6.1f} uM"
              f" | mass_err {d['mass_error']:.1e}{tag}")

    print("\n   The dimer (GlcNAc)2 accumulates because the GH20 chitobiase")
    print("   (0.024 s^-1 on chitobiose, Enterobacter, ref 10.1038/s41598-017-05140-3)")
    print("   is slow relative to the chitinases producing it.  Reported chitobiose")
    print("   kcat/Km spans ~5 orders across enzymes, so this rests on enzyme choice.")