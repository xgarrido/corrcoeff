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

def svd_pow(A, exponent):
    E, V = np.linalg.eigh(A)
    return np.einsum("...ab,...b,...cb->...ac",V,E**exponent,V)

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

def fisher(setup, covmat_params):
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    study = experiment["study"]

    from copy import deepcopy

    params = covmat_params
    covmat = setup.get("simulation").get("covmat")
    epsilon = 0.01
    deriv = {}
    for p in params:
        setup_mod = deepcopy(setup)
        parname = p if p != "logA" else "As"
        value = setup["simulation"]["cosmo. parameters"][parname]
        setup_mod["simulation"]["cosmo. parameters"][parname] = (1-epsilon)*value
        Cl_minus = get_theory_cls(setup_mod, lmax)
        setup_mod["simulation"]["cosmo. parameters"][parname] = (1+epsilon)*value
        Cl_plus = get_theory_cls(setup_mod, lmax)
        if study == "R":
            plus = Cl_plus["te"]/np.sqrt(Cl_plus["tt"]*Cl_plus["ee"])
            minus = Cl_minus["te"]/np.sqrt(Cl_minus["tt"]*Cl_minus["ee"])
        elif study == "TE":
            plus = Cl_plus["te"]
            minus = Cl_minus["te"]
        d = (plus[lmin:lmax]-minus[lmin:lmax])/(2*epsilon*value)
        deriv[p] = d if p != "logA" else d*value

    nparam = len(params)
    fisher = np.zeros((nparam,nparam))
    for count1, p1 in enumerate(params):
        for count2, p2 in enumerate(params):
            fisher[count1,count2] = np.sum(covmat**-1*deriv[p1]*deriv[p2])
    cov = np.linalg.inv(fisher)
    print("eigenvalues = ", np.linalg.eigvals(cov))
    for count, p in enumerate(params):
        if p == "logA":
            value = np.log(1e10*setup_mod["simulation"]["cosmo. parameters"]["As"])
        else:
            value = setup_mod["simulation"]["cosmo. parameters"][p]
        print(p, value, np.sqrt(cov[count,count]))
    return cov
