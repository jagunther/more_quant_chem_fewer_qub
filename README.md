# more_quant_chem_fewer_qub

A repository for reproducing the data and plots of https://arxiv.org/abs/2308.16873.


### B-H dissocation curve
Run `molecules/BH/dissociation_equilibrium.py` to compute energies (HF, FCI, CASCI(5,4), MRPT2, SC-NEVPT2)
and store them in the pickle file `BH_data_631G.pkl`. 
The computation takes about an hour on a single machine.
Figure 2 of the paper is created with `molecules/BH/plot_pes.py`.

### H2O2 
The three stationary points are optimised at the CASCI(10,14) level using the script
`molecules/H2O2/stationarypoint_opt.py`, the resulting xyz-files are stored in the same folder.
With `molecules/H2O2/pes_energies.py` the energies reported in Table II of the paper are computed,
which takes about an hour on a single machine.
The scaling of the perturbation norm for H2O2 at the TS configuration as presented in Figure 4 and
in Table III is computed with `molecules/H2O2/perturbation_norm.py`.

### Hydrogenchain
The scaling of the perturbation norm for the Hydrogenchain as presented in Figure 5 and in
Table III is computed with `molecules/hydrogenchain/perturbation_norm_chain.py`.

### Tests
The folder `test` contains some tests of the methods contained in the folder `more_quant_chem_fewer_qub`.



For questions, please send a mail to jmg@math.ku.dk or jakobguenther@hotmail.de.