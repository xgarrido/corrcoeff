import numpy as np

def get_theory_cls(setup, lmax, ell_factor=False):
    # Get simulation parameters
    simu = setup["simulation"]
    cosmo = simu["cosmo. parameters"]
    # CAMB use As
    if "logA" in cosmo:
        cosmo["As"] = 1e-10*np.exp(cosmo["logA"])
        del cosmo["logA"]

    # Get cobaya setup
    from copy import deepcopy
    info = deepcopy(setup["cobaya"])
    info["params"] = cosmo
    # Fake likelihood so far
    info["likelihood"] = {"one": None}
    from cobaya.model import get_model
    model = get_model(info)

    model.likelihood.theory.needs(Cl={"tt": lmax, "ee": lmax, "te": lmax})
    model.logposterior({}) # parameters are fixed
    Cls = model.likelihood.theory.get_cl(ell_factor=ell_factor)
    return Cls

def bin_spectrum(dl, l, lmin, lmax, delta_l):
    Nbin = np.int(lmax/delta_l)
    db = np.zeros(Nbin)
    lb = np.zeros(Nbin)
    for i in range(Nbin):
        idx = np.where((l> i*delta_l) & (l< (i+1)*delta_l))
        db[i] = np.mean(dl[idx])
        lb[i] = np.mean(l[idx])
    idx = np.where(lb>lmin)
    lb,db = lb[idx],db[idx]
    return lb, db

def bin_variance(vl, l, lmin, lmax, delta_l):
    Nbin = np.int(lmax/delta_l)
    vb = np.zeros(Nbin)
    lb = np.zeros(Nbin)
    for i in range(Nbin):
        idx = np.where((l>i*delta_l) & (l<(i+1)*delta_l))
        vb[i] = np.sum(1/vl[idx])
        lb[i] = np.mean(l[idx])
    vb=1/vb

    idx = np.where(lb>lmin)
    lb, vb = lb[idx], vb[idx]
    return lb, vb
