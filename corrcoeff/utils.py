import numpy as np

def bin_array(a, delta):
    return a.reshape(-1, delta).mean(axis=1)

def get_noise(setup, ell_factor=True):
    lmin, lmax = setup["lmin"], setup["lmax"]
    ls = np.arange(lmin, lmax)
    ell_factor = ls*(ls+1)/2/np.pi if ell_factor else 1.0
    use = setup["use"]
    if use == "SO":
        fsky = setup["fsky"]
        sensitivity_mode = setup[use]["sensitivity_mode"]
        from corrcoeff import V3calc as V3
        ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels \
            = V3.Simons_Observatory_V3_LA_noise(sensitivity_mode, fsky, lmin, lmax, delta_ell=1, apply_beam_correction=True)
        # Keep only relevant rows
        idx = np.intersect1d(setup[use]["freq"], setup[use]["freq_all"], return_indices=True)[-1]
        return ell_factor*N_ell_T_LA[idx], ell_factor*N_ell_P_LA[idx]
    elif use == "Planck":
        l = np.arange(lmin, lmax)
        beam_FWHM = np.array(setup[use]["beam_th"])
        beam = np.deg2rad(beam_FWHM)/60/np.sqrt(8*np.log(2))
        gaussian_beam = np.exp(l*(l+1)*beam[:, None]**2)
        sigma_th_temp = np.deg2rad(np.array(setup[use]["sigma_th_temp"]))/60
        sigma_th_polar = np.deg2rad(np.array(setup[use]["sigma_th_polar"]))/60
        N_ell_T = sigma_th_temp[:, None]**2*gaussian_beam
        N_ell_P = sigma_th_polar[:, None]**2*gaussian_beam
        return ell_factor*N_ell_T, ell_factor*N_ell_P
    else:
        raise ValueError("Unkown experiment '{}".format(use))

def get_systematics(setup):
    lmin, lmax = setup["lmin"], setup["lmax"]
    l = np.arange(lmin, lmax)

    def _parse(str):
        try:
            return float(str)
        except:
            str = str.strip()
            if str.endswith("%"):
                return float(str.strip("%").strip()) / 100
            raise Exception("Don't know how to parse %s" % str)

    systematics = setup["systematics"]
    syst_beam = 1 - _parse(systematics.get("beam", 0.0))
    syst_polar = 1 - _parse(systematics.get("polar", 0.0))

    FWHM_fid= 1.5
    beam_FWHM_rad_fid = np.deg2rad(FWHM_fid)/60
    beam_fid = beam_FWHM_rad_fid/np.sqrt(8*np.log(2))
    bl_fid = np.exp(-l*(l+1)*beam_fid**2/2.)
    beam_FWHM_rad_syst = np.deg2rad(syst_beam*FWHM_fid)/60
    beam_syst = beam_FWHM_rad_syst/np.sqrt(8*np.log(2))
    bl_syst = np.exp(-l*(l+1)*beam_syst**2/2.)

    syst_beam = (bl_fid/bl_syst)**2

    # Transfer function TF[T, E]
    min_TF = np.array([[systematics.get("TF").get("min_T", 1.0)],
                       [systematics.get("TF").get("min_E", 1.0)]])
    lmax = np.array([[systematics.get("TF").get("lmax_T", 0)],
                     [systematics.get("TF").get("lmax_E", 0)]])
    TF = np.where(l < lmax, min_TF + (1 - min_TF)*(np.cos((lmax - l)/(lmax - 2)*np.pi/2))**2, 1.0)

    TT_syst = syst_beam*TF[0]**2
    TE_syst = syst_beam*syst_polar*TF[0]*TF[1]
    EE_syst = syst_beam*syst_polar**2*TF[1]**2

    return TT_syst, TE_syst, EE_syst


def get_theory_cls(setup, lmax, ell_factor=True):
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
    Cls = model.likelihood.theory.get_Cl(ell_factor=ell_factor)

    pars = model.likelihood.theory.camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, tau=0.06)
    pars.set_for_lmax(lmax,lens_potential_accuracy=1)
    print("before", pars)
    # if simu.get("Pk", None):
    #     print("Have Pk")
    #     if not info.get("theory").get("camb", None):
    #         raise Exception("Power spectra modification can only be done with 'CAMB'!")

    #     from cobaya.tools import get_external_function
    #     Pk = get_external_function(simu["Pk"])
    #     pars.set_initial_power_function(Pk,
    #                                     args=(cosmo["As"], cosmo["ns"], 0.0599, 280, 0.08, 0.2, 0),
    #                                     effective_ns_for_nonlinear=cosmo["ns"])


    pars.InitPower.set_params(As=cosmo["As"], ns=cosmo["ns"])
    results = model.likelihood.theory.camb.get_results(pars)
    cl = results.get_lensed_scalar_cls(CMB_unit ='muK')
    import pickle
    pickle.dump({"Cl": cl}, open("cl.pkl", "wb"))


    return Cls

def fisher(setup, covmat_params):
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    study = experiment["study"]
    fsky = experiment["fsky"]
    ls = np.arange(lmin, lmax)
    nell = np.alen(ls)

    from copy import deepcopy

    params = covmat_params
    epsilon = 0.01
    if "joint" in study:
        deriv = np.empty((len(params), 2, 2, nell))
    else:
        deriv = np.empty((len(params), nell))

    for i, p in enumerate(params):
        setup_mod = deepcopy(setup)
        parname = p if p != "logA" else "As"
        value = setup["simulation"]["cosmo. parameters"][parname]
        setup_mod["simulation"]["cosmo. parameters"][parname] = (1-epsilon)*value
        Cl_minus = get_theory_cls(setup_mod, lmax)
        setup_mod["simulation"]["cosmo. parameters"][parname] = (1+epsilon)*value
        Cl_plus = get_theory_cls(setup_mod, lmax)

        d = {}
        for s in ["tt", "te", "ee", "r"]:
            if s == "r":
                plus = Cl_plus["te"]/np.sqrt(Cl_plus["tt"]*Cl_plus["ee"])
                minus = Cl_minus["te"]/np.sqrt(Cl_minus["tt"]*Cl_minus["ee"])
            else:
                plus, minus = Cl_plus[s], Cl_minus[s]
            delta = (plus[lmin:lmax] - minus[lmin:lmax])/(2*epsilon*value)
            d[s] = delta if p != "logA" else delta*value

        if "joint" in study:
            deriv[i] = np.array([[d["tt"], d["te"]],
                                 [d["te"], d["ee"]]])
        else:
            deriv[i] = d[study.lower()]

    # Compute covariance matrix
    if experiment.get("add_noise"):
        # Get SO noise
        N_TT, N_EE = get_noise(experiment)
        N_TT, N_EE = 1/np.sum(1/N_TT, axis=0), 1/np.sum(1/N_EE, axis=0)
    else:
        N_TT = 0.0
        N_EE = 0.0

    Cls = get_theory_cls(setup, lmax)
    Cl_TT = Cls["tt"][lmin:lmax]
    Cl_TE = Cls["te"][lmin:lmax]
    Cl_EE = Cls["ee"][lmin:lmax]
    if "joint" in study:
        C = np.array([[Cl_TT + N_TT, Cl_TE],
                      [Cl_TE, Cl_EE + N_EE]])
    elif study == "TT":
        C = 2*(Cl_TT + N_TT)**2
    elif study == "TE":
        C = (Cl_TT + N_TT)*(Cl_EE + N_EE) + Cl_TE**2
    elif study == "EE":
        C = 2*(Cl_EE + N_EE)**2
    elif study == "R":
        R = Cl_TE/np.sqrt(Cl_TT*Cl_EE)
        C = R**4 - 2*R**2 + 1 + N_TT/Cl_TT + N_EE/Cl_EE + (N_TT*N_EE)/(Cl_TT*Cl_EE) \
            + R**2*(0.5*(N_TT/Cl_TT - 1)**2 + 0.5*(N_EE/Cl_EE - 1)**2 - 1)

    inv_C = C**-1
    if "joint" in study:
        for l in range(nell):
            inv_C[:,:,l] = np.linalg.inv(C[:,:,l])

    # Fisher matrix
    nparam = len(params)
    fisher = np.empty((nparam,nparam))
    for p1 in range(nparam):
        for p2 in range(nparam):
            somme = 0.0
            if "joint" in study:
                for l in range(nell):
                    m1 = np.dot(inv_C[:,:,l], deriv[p1,:,:,l])
                    m2 = np.dot(inv_C[:,:,l], deriv[p2,:,:,l])
                    somme += (2*ls[l]+1)/2*fsky*np.trace(np.dot(m1, m2))
            else:
                somme = np.sum((2*ls+1)*fsky*inv_C*deriv[p1]*deriv[p2])
            fisher[p1, p2] = somme

    cov = np.linalg.inv(fisher)
    print("eigenvalues = ", np.linalg.eigvals(cov))
    for count, p in enumerate(params):
        if p == "logA":
            value = np.log(1e10*setup_mod["simulation"]["cosmo. parameters"]["As"])
        else:
            value = setup_mod["simulation"]["cosmo. parameters"][p]
        print(p, value, np.sqrt(cov[count,count]))
    return cov
