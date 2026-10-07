"""
A MECHANISTIC MODEL OF ENZYMATIC CELLULOSE DEGRADATION
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import poisson

# PARAMETERS
@dataclass
class Parameters:
    # substrate-microcrystalline cellulose (Avicel) 
    dp_initial:        float = 300.0     
    dp_max:            int   = 350      
    loading_g_per_L:   float = 10.0      
    surface_area_m2_g: float = 47.6      
    density_g_per_L:   float = 1500.0    
    mw_glucose_g_mmol: float = 0.162     
    kcat_EG_per_h:   float = 65.0 * 3600.0    
    kcat_CBH_per_h:  float = 9.4  * 3600.0   
    KM_EG_cellulose: float = 0.0067          
    KM_CBH_cellulose:float = 1.4e-5           
    KM_EG_soluble:   float = 0.053            
    KM_CBH_soluble:  float = 0.0032           
    # competitive product inhibition 
    Ki_EG_glucose:    float = 16.9   
    Ki_EG_cellobiose: float = 0.01   
    Ki_CBH_glucose:   float = 31.0   
    Ki_CBH_cellobiose:float = 0.093  
    # adsorption and surface crowding 
    k_ads_L_mmol_h:  float = 8640.0  
    k_des_EG_per_h:  float = 19.3    
    k_des_CBH_per_h: float = 164.0   
    enzyme_footprint:float = 156.0   
    max_site_density:float = 6.64e-5 
    # thermal deactivation half-lives
    half_life_EG_h:  float = 4.3    
    half_life_CBH_h: float = 10.6  
    half_life_BG_h:  float = 24.0    
    # enzyme loadings
    EG_total_mmol_L:  float = 1.6e-3 
    CBH_total_mmol_L: float = 1.6e-3 
    BG_total_mmol_L:  float = 0.16e-3 
    # beta-glucosidase 
    kcat_BG_per_h:   float = 26.6 * 3600.0   
    KM_BG_mmol_L:    float = 0.33            
    kcat_BGox_per_h: float = 0.3 * 26.6*3600 
    KM_BGox_mmol_L:  float = 0.5             
    cellulase_efficiency: float = 1.0       

    # LPMO catalytic cycle 
    LPMO_total_uM:      float = 0.5     
    kcat_LPMO_per_h:    float = 8.5 * 3600.0  
    KM_H2O2_uM:         float = 30.0    
    S_half_Avicel_g_L:  float = 13.0     
    futile_efficiency_ratio: float = 0.10    
    lpmo_protected_fraction: float = 0.98     
    p_inactivation_per_futile: float = 0.0072  
    H2O2_supply_uM_h:   float = 83.4     
    ascorbate_init_uM:  float = 100.0    
    H2O2_init_uM:       float = 0.0      
    k_priming_per_uM_h: float = 1.0      
    k_reoxidation_per_h:float = 0.5     
    k_ascorbate_autox_per_h:   float = 0.02  
    lpmo_C1_fraction:   float = 1.0      
    cbh_activity_on_C1ox: float = 0.68   

class Model:
    def __init__(self, p: Parameters):
        self.p = p
        self.dp = np.arange(7, p.dp_max + 1)          
        self.n_solid = self.dp.size

        n = self.n_solid
        self.I_solid    = slice(0, n)
        self.I_soluble  = slice(n, n + 6)
        self.i_gluconic    = n + 6      
        self.i_cellobionic = n + 7      
        self.i_oligo_ox    = n + 8      
        self.i_ends_ox     = n + 9      
        self.i_CuII        = n + 10
        self.i_CuI         = n + 11
        self.i_Cudead      = n + 12
        self.i_H2O2        = n + 13
        self.i_ascorbate   = n + 14
        self.i_EG_ads      = n + 15
        self.i_CBH_ads     = n + 16
        self.i_radius      = n + 17
        self.n_state       = n + 18

        # total glucose in the system (mmol/L)
        self.total_glucose = p.loading_g_per_L / p.mw_glucose_g_mmol

        # initial solid chain-length distribution ~ Poisson(DP0), normalised
        self.chain_pmf = poisson.pmf(self.dp, p.dp_initial)
        self.chain_pmf /= self.chain_pmf.sum()

        self._setup_geometry()

    def _setup_geometry(self):
        p = self.p
        sa_dm2_g = p.surface_area_m2_g * 100.0                 
        self.radius0 = 3.0 / (sa_dm2_g * p.density_g_per_L)    
        area_per_L0  = sa_dm2_g * p.loading_g_per_L            
        self.n_particles = area_per_L0 / (4.0*np.pi*self.radius0**2)  

    def area_per_volume(self, radius):
        #Cellulose surface area per litre of liquid 
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
        glucose, cellobiose = soluble[0], soluble[1]
        inhib_EG_sol = (np.sum((np.arange(3, 7) - 1.0)*soluble[2:6]) / p.KM_EG_soluble
                        + glucose/p.Ki_EG_glucose + cellobiose/p.Ki_EG_cellobiose)
        inhib_CBH_sol = (np.sum(soluble[2:6]) / p.KM_CBH_soluble
                         + glucose/p.Ki_CBH_glucose + cellobiose/p.Ki_CBH_cellobiose)
        return inhib_EG_sol, inhib_CBH_sol

    #  Cellulose site balance  
    def site_balance(self, EG_ads, CBH_ads, solid, soluble):
        p = self.p
        occ_EG  = p.enzyme_footprint * EG_ads  / p.max_site_density
        occ_CBH = p.enzyme_footprint * CBH_ads / p.max_site_density
        bonds_solid  = np.sum((self.dp - 1.0) * solid)    
        chains_solid = np.sum(solid)                       
        inhib_EG_sol, inhib_CBH_sol = self._inhibition_terms(soluble)

        a = (1.0
             - occ_EG *(1.0 + occ_EG *bonds_solid /p.KM_EG_cellulose  + inhib_EG_sol)
             - occ_CBH*(1.0 + occ_CBH*chains_solid/p.KM_CBH_cellulose + inhib_CBH_sol))
        b = (occ_EG *bonds_solid /p.KM_EG_cellulose
             + occ_CBH*chains_solid/p.KM_CBH_cellulose)
        theta = a / (1.0 + b)
        theta = min(max(theta, 0.0), 1.0)                  
        return theta, min(max(theta+occ_EG,0.0),1.0), min(max(theta+occ_CBH,0.0),1.0)

    #  LPMO four-stage catalytic cycle 
    def lpmo_cycle(self, CuII, CuI, H2O2, ascorbate, solid_conc_g_L):

        p = self.p
        accessibility = solid_conc_g_L / (solid_conc_g_L + p.S_half_Avicel_g_L)   
        v_priming = p.k_priming_per_uM_h * CuII * ascorbate
        v_peroxygenase = (p.kcat_LPMO_per_h * CuI * H2O2
                          / (p.KM_H2O2_uM + H2O2)) * accessibility
        free_frac = (1.0 - p.lpmo_protected_fraction)
        v_red_perox = (p.futile_efficiency_ratio * p.kcat_LPMO_per_h * CuI * H2O2
                       / (p.KM_H2O2_uM + H2O2)) * free_frac
        v_inactivation = p.p_inactivation_per_futile * v_red_perox
        v_reoxidation  = p.k_reoxidation_per_h * CuI

        dCuII = -v_priming + v_reoxidation
        dCuI  = +v_priming - v_reoxidation - v_inactivation   
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
        cellobionic  = max(y[self.i_cellobionic], 0.0)
        ends_ox      = max(y[self.i_ends_ox],     0.0)
        CuII = max(y[self.i_CuII], 0.0); CuI = max(y[self.i_CuI], 0.0)
        H2O2 = max(y[self.i_H2O2], 0.0); ascorbate = max(y[self.i_ascorbate], 0.0)
        EG_ads  = max(y[self.i_EG_ads],  0.0)
        CBH_ads = max(y[self.i_CBH_ads], 0.0)
        radius  = max(y[self.i_radius], 1e-12)
        cellobiose = soluble[1]

        area_vol = self.area_per_volume(radius)                 

        # thermal deactivation of the hydrolytic enzymes
        decay = (lambda h: np.exp(-np.log(2)*t/h)) if thermal_decay else (lambda h: 1.0)
        EG_tot  = p.EG_total_mmol_L  * decay(p.half_life_EG_h)
        CBH_tot = p.CBH_total_mmol_L * decay(p.half_life_CBH_h)
        BG_tot  = p.BG_total_mmol_L  * decay(p.half_life_BG_h)

        # site balance and free enzyme 
        theta, theta_EG, theta_CBH = self.site_balance(EG_ads, CBH_ads, solid, soluble)
        free_sites = p.max_site_density * theta
        inhib_EG_sol, inhib_CBH_sol = self._inhibition_terms(soluble)
  
        bracket_EG  = (1.0 + theta_EG *np.sum((dp-1.0)*solid)/p.KM_EG_cellulose
                       + inhib_EG_sol)
        bracket_CBH = (1.0 + theta_CBH*np.sum(solid) /p.KM_CBH_cellulose
                       + inhib_CBH_sol)
        EG_free  = max(EG_tot  - area_vol*EG_ads *bracket_EG,  0.0)  
        CBH_free = max(CBH_tot - area_vol*CBH_ads*bracket_CBH, 0.0)   

        # adsorption ODEs 
        dEG_ads  = p.k_ads_L_mmol_h*EG_free *free_sites - p.k_des_EG_per_h *EG_ads
        dCBH_ads = p.k_ads_L_mmol_h*CBH_free*free_sites - p.k_des_CBH_per_h*CBH_ads

        # per-bond / per-end catalytic rate constants 
        s = p.cellulase_efficiency
        eg_scission     = s*(p.kcat_EG_per_h/p.KM_EG_cellulose)  *EG_ads *theta_EG   
        cbh_removal_nat = s*(p.kcat_CBH_per_h/p.KM_CBH_cellulose)*CBH_ads*theta_CBH  

        # substrate-depletion guard 
        depl = min(max(radius/(1.0e-3*self.radius0), 0.0), 1.0)
        eg_scission     *= depl
        cbh_removal_nat *= depl

        # CBH impediment on C1-oxidised reducing ends (Keller 2021) 
        total_ends_vol = np.sum(solid)*area_vol                
        frac_ox = min(ends_ox/total_ends_vol, 1.0) if total_ends_vol > 1e-30 else 0.0
        phi = p.cbh_activity_on_C1ox
        cbh_factor  = (1.0 - frac_ox) + phi*frac_ox             
        cbh_removal = cbh_removal_nat * cbh_factor
        
        ox_turnover_share = (phi*frac_ox/cbh_factor) if cbh_factor > 1e-30 else 0.0

        #  LPMO cycle -> oxidative-cut rate
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
        eg_term   = eg_scission   * (2.0*cum_longer - (dp-1.0)*solid)   
        lpmo_term = lpmo_scission * (2.0*cum_longer - (dp-1.0)*solid)   
        cbh_term  = cbh_removal   * (solid_plus2 - solid)              
        dsolid_reaction = eg_term + lpmo_term + cbh_term

        # soluble fragments released from the solid (per area) 
      
        longer_than = np.array([np.sum(solid[dp > k]) for k in range(0, 7)])
        native_release   = np.zeros(7)
        oxidised_release = np.zeros(7)
        f_C1 = p.lpmo_C1_fraction
        for k in range(1, 7):
            native_release[k]   = (2.0*eg_scission*longer_than[k]
                                   + (2.0 - f_C1)*lpmo_scission*longer_than[k])
            oxidised_release[k] = f_C1*lpmo_scission*longer_than[k]
        # CBH releases cellobiose from every processive turnover; DP7->C5, DP8->C6
        cbh_cellobiose = cbh_removal*np.sum(solid)
        native_release[5] += cbh_removal*solid[0]
        native_release[6] += cbh_removal*solid[1]

        cbh_cellobiose_vol   = cbh_cellobiose*area_vol          
        cellobionic_from_cbh = ox_turnover_share*cbh_cellobiose_vol
        native_cellobiose_cbh= (1.0-ox_turnover_share)*cbh_cellobiose_vol

        # total glucose leaving the solid this instant (for the radius, Eq.1)
        areal_glucose_flux = (np.sum([k*native_release[k]   for k in range(1,7)])
                              + np.sum([k*oxidised_release[k] for k in range(1,7)])
                              + 2.0*cbh_cellobiose)

        # new-chain exposure as the particle recedes 
        dsolid_exposure = self.chain_pmf * (areal_glucose_flux/p.dp_initial)
        dsolid = dsolid_reaction + dsolid_exposure

        #  Soluble-phase balances 
        d_soluble = np.zeros(6)
        for k in range(1, 7):
            d_soluble[k-1] += native_release[k]*area_vol
        d_soluble[1] += native_cellobiose_cbh                    
        # soluble-phase CBH:  C_k -> C_{k-2} + cellobiose  (adsorbed + free enzyme)
        cbh_soluble = (CBH_free + CBH_ads*area_vol)*(p.kcat_CBH_per_h/p.KM_CBH_soluble)
        for k in (6, 5, 4, 3):
            v = cbh_soluble*soluble[k-1]
            d_soluble[k-1] -= v; d_soluble[k-3] += v; d_soluble[1] += v
        # soluble-phase EG random scission of C3..C6 
        eg_soluble = (EG_free + EG_ads*area_vol)*(p.kcat_EG_per_h/p.KM_EG_soluble)
        for k in (6, 5, 4, 3):
            per_bond = eg_soluble*soluble[k-1]
            d_soluble[k-1] -= (k-1.0)*per_bond
            for cut in range(1, k):
                d_soluble[cut-1]     += per_bond
                d_soluble[k-cut-1]   += per_bond
        v_BG = (p.kcat_BG_per_h*BG_tot)*cellobiose/(p.KM_BG_mmol_L + cellobiose)
        d_soluble[1] -= v_BG; d_soluble[0] += 2.0*v_BG
        # oxidised product pools
        d_gluconic    = oxidised_release[1]*area_vol           
        d_cellobionic = oxidised_release[2]*area_vol + cellobionic_from_cbh
        d_oligo_ox    = np.sum([k*oxidised_release[k] for k in range(3,7)])*area_vol
        v_ox_hydrolysis = (p.kcat_BGox_per_h*BG_tot)*cellobionic/(p.KM_BGox_mmol_L + cellobionic)
        d_cellobionic -= v_ox_hydrolysis
        d_soluble[0]  += v_ox_hydrolysis                      
        d_gluconic    += v_ox_hydrolysis                         
        # C1-oxidised reducing ends on the solid 
        d_ends_ox = 0.0
        if use_lpmo:
            ends_created  = lpmo_cut_rate * p.lpmo_C1_fraction
            ends_released = np.sum(oxidised_release[1:7])*area_vol
            d_ends_ox = ends_created - ends_released - cellobionic_from_cbh

        # particle radius 
        d_radius = -(p.mw_glucose_g_mmol/p.density_g_per_L)*areal_glucose_flux

        # assemble derivative vector 
        dydt = np.zeros(self.n_state)
        dydt[self.I_solid]   = dsolid
        dydt[self.I_soluble] = d_soluble
        dydt[self.i_gluconic]    = d_gluconic
        dydt[self.i_cellobionic] = d_cellobionic
        dydt[self.i_oligo_ox]    = d_oligo_ox
        dydt[self.i_ends_ox]     = d_ends_ox
        dydt[self.i_CuII]  = dCuII; dydt[self.i_CuI] = dCuI; dydt[self.i_Cudead] = dCudead
        dydt[self.i_H2O2]  = dH2O2; dydt[self.i_ascorbate] = dAsc
        dydt[self.i_EG_ads]  = dEG_ads
        dydt[self.i_CBH_ads] = dCBH_ads
        dydt[self.i_radius]  = d_radius
        return dydt

    #  Integrate                                                        
    
    def _atol_vector(self):
        atol = np.empty(self.n_state)
        atol[self.I_solid]   = 1e-12          
        atol[self.I_soluble] = 1e-8          
        atol[self.i_gluconic:self.i_ends_ox+1] = 1e-8  
        atol[self.i_CuII:self.i_H2O2+1] = 1e-6         
        atol[self.i_ascorbate] = 1e-4                  
        atol[self.i_EG_ads]  = 1e-12
        atol[self.i_CBH_ads] = 1e-12
        atol[self.i_radius]  = 1e-14
        return atol

    def simulate(self, use_lpmo=False, thermal_decay=True, t_end_h=12.0,
                 n_points=200, max_step=np.inf):
        # BDF-the coupled adsorption/site-balance system is stiff. Matches the validated hydrolytic baseline (BDF, rtol 1e-6).
        self._setup_geometry()
        sol = solve_ivp(self.rhs, (0.0, t_end_h), self.initial_state(),
                        args=(use_lpmo, thermal_decay), method="BDF",
                        rtol=1e-6, atol=self._atol_vector(), max_step=max_step,
                        t_eval=np.linspace(0.0, t_end_h, n_points))
        return sol

    #  Post-processing                                                   #

    def analyse(self, sol):
        r = sol.y[self.i_radius]
        soluble = sol.y[self.I_soluble]
        gluconic    = sol.y[self.i_gluconic]
        cellobionic = sol.y[self.i_cellobionic]
        oligo_ox    = sol.y[self.i_oligo_ox]
        # solid glucose from the particle volume (independent of the soluble pools)
        solid_glucose = (4.0/3.0)*np.pi*r**3*self.p.density_g_per_L*self.n_particles/self.p.mw_glucose_g_mmol
        native_soluble = np.sum([(k+1)*soluble[k] for k in range(6)], axis=0)
        oxidised_soluble = 1.0*gluconic + 2.0*cellobionic + oligo_ox        
        total = solid_glucose + native_soluble + oxidised_soluble
        return dict(
            t=sol.t, radius=r, solid_glucose=solid_glucose,
            glucose=soluble[0], cellobiose=soluble[1], oligomers=soluble[2:].sum(0),
            cellotriose=soluble[2], cellotetraose=soluble[3],
            cellopentaose=soluble[4], cellohexaose=soluble[5],
            soluble_all=soluble,
            gluconic_acid=gluconic, cellobionic_acid=cellobionic, oxidised_oligomers=oligo_ox,
            conversion_pct=100.0*(1.0 - solid_glucose/self.total_glucose),
            total_glucose=total, mass_error=np.max(np.abs(total - self.total_glucose)),
            CuII=sol.y[self.i_CuII], CuI=sol.y[self.i_CuI], Cu_dead=sol.y[self.i_Cudead],
            H2O2=sol.y[self.i_H2O2], ascorbate=sol.y[self.i_ascorbate],
            ends_ox=sol.y[self.i_ends_ox],
            h2o2_valid_until_h=self._h2o2_validity(sol))

    def _h2o2_validity(self, sol, km_multiple=1.0):
        km_h2o2_uM = self.p.KM_H2O2_uM * km_multiple
        h = sol.y[self.i_H2O2]
        idx = np.argmax(h >= km_h2o2_uM)
        return float(sol.t[idx]) if h[idx] >= km_h2o2_uM else None

    def site_traces(self, sol, use_lpmo):
        #Recover theta_EG, theta_CBH and separate EG- vs LPMO-driven new-end rates.
        p = self.p
        t = sol.t
        theta_EG = np.zeros_like(t); theta_CBH = np.zeros_like(t)
        ends_from_EG = np.zeros_like(t); ends_from_LPMO = np.zeros_like(t)
        accessible_ends = np.zeros_like(t)
        for i in range(t.size):
            solid = np.maximum(sol.y[self.I_solid, i], 0.0)
            soluble = np.maximum(sol.y[self.I_soluble, i], 0.0)
            EG_ads = max(sol.y[self.i_EG_ads, i], 0.0); CBH_ads = max(sol.y[self.i_CBH_ads, i], 0.0)
            radius = max(sol.y[self.i_radius, i], 1e-12)
            tEG, tCBH = self.site_balance(EG_ads, CBH_ads, solid, soluble)
            theta_EG[i] = tEG; theta_CBH[i] = tCBH
            area_vol = self.area_per_volume(radius)
            eg_scission = p.cellulase_efficiency*(p.kcat_EG_per_h/p.KM_EG_cellulose)*EG_ads*tEG
            ends_from_EG[i] = 2.0*eg_scission*np.sum(solid)*area_vol*1e3   
            accessible_ends[i] = np.sum(solid)*area_vol*1e3                 
            if use_lpmo:
                CuI = max(sol.y[self.i_CuI, i], 0.0); H2O2 = max(sol.y[self.i_H2O2, i], 0.0)
                s_g_L = p.loading_g_per_L*(radius/self.radius0)**3
                acc = s_g_L/(s_g_L + p.S_half_Avicel_g_L)
                cut = (p.kcat_LPMO_per_h*CuI*H2O2/(p.KM_H2O2_uM+H2O2))*acc*1e-3  
                ends_from_LPMO[i] = cut*1e3                                
        return dict(t=t, theta_EG=theta_EG, theta_CBH=theta_CBH,
                    ends_from_EG=ends_from_EG, ends_from_LPMO=ends_from_LPMO,
                    accessible_ends=accessible_ends)

# CHECK

if __name__ == "__main__":
    import numpy as _np

    print("1) REDUCTION TO THE VALIDATED HYDROLYTIC BASELINE")
    print("   (BG = 0, LPMO off, thermal off; standalone Levine baseline gives 22.37 %)")
    for sa, ref in ((8.0, 6.34), (47.6, 22.37)):
        m = Model(Parameters(BG_total_mmol_L=0.0, surface_area_m2_g=sa))
        d = m.analyse(m.simulate(use_lpmo=False, thermal_decay=False, n_points=80))
        print(f"   SA={sa:5.1f} m2/g : conversion@12h = {d['conversion_pct'][-1]:6.2f} %   "
              f"(baseline {ref:5.2f} %)   mass_err = {d['mass_error']:.1e}")

    print("\n2) FULL SYSTEM (EG + CBH + BG, thermal deactivation ON)")
    for lpmo in (False, True):
        m = Model(Parameters())
        d = m.analyse(m.simulate(use_lpmo=lpmo, n_points=200))
        c = d['conversion_pct']
        i99 = _np.argmax(c >= 99.0)
        t99 = d['t'][i99] if c[i99] >= 99.0 else None
        line = (f"   LPMO={str(lpmo):5s} conv@12h={c[-1]:6.2f}%  mass_err={d['mass_error']:.1e}"
                f"  t(99%)={t99:.2f} h" if t99 else
                f"   LPMO={str(lpmo):5s} conv@12h={c[-1]:6.2f}%  mass_err={d['mass_error']:.1e}")
        print(line)
        if lpmo:
            v = d['h2o2_valid_until_h']
            print(f"          Cu dead@48h = {100*d['Cu_dead'][-1]/Parameters().LPMO_total_uM:.1f} %"
                  f" | gluconic = {d['gluconic_acid'][-1]:.3f} mmol/L")
            if v is not None:
                km = Parameters().KM_H2O2_uM
                print(f"[H2O2] reaches Km(H2O2) = {km:.0f} uM at t = {v:.2f} h; beyond")
                print(f"this the LPMO is >half-saturated in co-substrate and the system")
                print(f"moves from the peroxide-limited to the enzyme-limited regime.")
