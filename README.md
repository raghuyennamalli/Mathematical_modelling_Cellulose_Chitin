Mathematical modelling of cellulose and chitin degradation in the presence and absence of lytic polysaccharide monooxygenases

This repository contains Python scripts for mathematical modeling of enzymatic cellulose and chitin degradation using a shrinking-particle framework, with and without lytic polysaccharide monooxygenase (LPMO)-mediated oxidative cleavage.

File Description
cellulose_model.py-Mathematical model for cellulose degradation by endoglucanase (EG), cellobiohydrolase (CBH), and beta-glucosidase (BG), with an LPMO module.
chitin_model.py-Mathematical model for chitin degradation by chitinases and beta-N-acetylhexosaminidase, with an LPMO module.
cellulose_sensitivity.py-Local sensitivity analysis of the cellulose model using +/-20% parameter perturbations and a central-difference calculation.
chitin_sensitivity.py-Local sensitivity analysis of the chitin model using +/-20% parameter perturbations and a central-difference calculation.
figures_cellulose_model.py-Script for generating cellulose model result figures.
figures_chitin_model.py-Script for generating chitin model result figures.

Model overview
The framework represents cellulose and chitin as shrinking spherical particles whose accessible surface area changes during degradation. The model accounts for:
- substrate particle geometry and surface-area changes
- polymer-chain length distributions
- enzyme adsorption and desorption
- surface-site availability
- hydrolytic cleavage by substrate-specific enzymes
- soluble oligomer and product formation
- downstream conversion of soluble cellulose products to glucose
- downstream conversion of soluble chitin products to GlcNAc and
- LPMO-mediated H2O2-dependent oxidative cleavage
The same mechanistic structure is used for both polysaccharides, while substrate and enzyme-specific parameters represent their different degradation behaviours.

LPMO activity is represented explicitly through H2O2-dependent peroxygenase chemistry rather than as an empirical increase in the hydrolysis rate.
This includes:
1. Cu(II)/Cu(I) redox cycling
2. H2O2-dependent LPMO turnover
3. substrate accessibility
4. productive oxidative cleavage
5. uncoupled H2O2 consumption
6. LPMO inactivation
7. substrate protection and
8. formation of oxidized soluble products.
   
The H2O2 supply is represented as a lumped input to the model.

Simulation conditions
The models can be evaluated under two conditions:
- -LPMO: LPMO terms are disabled to represent hydrolytic degradation alone.
- +LPMO: H2O2-dependent LPMO activity is included together with hydrolysis.
This allows the contribution of LPMO-mediated oxidative cleavage to be compared within the same framework.

Sensitivity analysis
The sensitivity scripts perform a local sensitivity analysis using a +/-20% perturbation around each nominal parameter value and a central finite-difference calculation.
The response variable is the predicted substrate conversion at 12 h.

The normalized sensitivity coefficient is:
S = (dY/Y) / (dP/P)

Reference: 
A Mechanistic Model of the Enzymatic Hydrolysis of Cellulose. Levine et.al (2010), Biotechnology and Bioengineering.
