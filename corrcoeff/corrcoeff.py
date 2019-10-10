# Global
import numpy as np

def simulation(setup):
    """
    Simulate CMB power spectrum given a set of cosmological parameters and noise level.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    delta = experiment.get("delta", None)
    fsky = experiment["fsky"]

    from corrcoeff import utils
    Dls = utils.get_theory_cls(setup, lmax)
    ls = np.arange(lmin, lmax)
    Dl_TT = Dls["tt"][lmin:lmax]
    Dl_TE = Dls["te"][lmin:lmax]
    Dl_EE = Dls["ee"][lmin:lmax]

    if experiment.get("systematics_file"):
        syst = np.loadtxt(experiment["systematics_file"])
        syst = syst[:,-1][lmin:lmax]
        Dl_TT *= syst
        Dl_TE *= syst
        Dl_EE *= syst
    elif experiment.get("systematics"):
        syst = utils.get_systematics(experiment)
        Dl_TT *= syst[0]
        Dl_TE *= syst[1]
        Dl_EE *= syst[2]

    if experiment.get("add_noise"):
        # Get SO noise
        N_TT, N_EE = utils.get_noise(experiment)
        N_TT, N_EE = 1/np.sum(1/N_TT, axis=0), 1/np.sum(1/N_EE, axis=0)
    else:
        N_TT = np.full_like(ls, 0.0)
        N_EE = np.full_like(ls, 0.0)

    if delta:
        ls = utils.bin_array(ls, delta)
        Dl_TT = utils.bin_array(Dl_TT, delta)
        Dl_TE = utils.bin_array(Dl_TE, delta)
        Dl_EE = utils.bin_array(Dl_EE, delta)
        N_TT = utils.bin_array(N_TT, delta)
        N_EE = utils.bin_array(N_EE, delta)

    R = Dl_TE/np.sqrt(Dl_TT*Dl_EE)
    covmat_RR   = R**4 - 2*R**2 + 1 + N_TT/Dl_TT + N_EE/Dl_EE + (N_TT*N_EE)/(Dl_TT*Dl_EE) \
        + R**2*(0.5*(N_TT/Dl_TT - 1)**2 + 0.5*(N_EE/Dl_EE - 1)**2 - 1)
    covmat_TTTT = 2*(Dl_TT+N_TT)**2
    covmat_TETE = (Dl_TT+N_TT)*(Dl_EE+N_EE) + Dl_TE**2
    covmat_EEEE = 2*(Dl_EE+N_EE)**2

    study = experiment["study"]
    if study == "R":
        # Compute TE correlation factor
        covmat = 1/(2*ls+1)/fsky*covmat_RR
        Dl_obs = R + np.sqrt(covmat)*np.random.randn(len(ls))
    elif study == "TT":
        covmat = 1/(2*ls+1)/fsky*covmat_TTTT
        Dl_obs = Dl_TT + np.sqrt(covmat)*np.random.randn(len(ls))
    elif study == "TE":
        covmat = 1/(2*ls+1)/fsky*covmat_TETE
        Dl_obs = Dl_TE + np.sqrt(covmat)*np.random.randn(len(ls))
    elif study == "EE":
        covmat = 1/(2*ls+1)/fsky*covmat_EEEE
        Dl_obs = Dl_EE + np.sqrt(covmat)*np.random.randn(len(ls))
    elif "joint" in study:
        covmat_TTEE = 2*Dl_TE**2
        covmat_TTTE = 2*(Dl_TT+N_TT)*Dl_TE
        covmat_TEEE = 2*(Dl_EE+N_EE)*Dl_TE
        covmat_REE  = R*(covmat_TEEE/Dl_TE - 0.5*covmat_EEEE/Dl_EE - 0.5*covmat_TTEE/Dl_TT)
        covmat_RTT  = R*(covmat_TTTE/Dl_TE - 0.5*covmat_TTEE/Dl_EE - 0.5*covmat_TTTT/Dl_TT)

        covmat = np.empty((3, 3, len(ls)))
        covmat[0,0,:] = covmat_TTTT
        covmat[0,1,:] = covmat_TTTE
        covmat[0,2,:] = covmat_TTEE
        covmat[1,1,:] = covmat_TETE
        covmat[1,2,:] = covmat_TEEE
        covmat[2,2,:] = covmat_EEEE

        covmat[1,0,:] = covmat[0,1,:]
        covmat[2,0,:] = covmat[0,2,:]
        covmat[2,1,:] = covmat[1,2,:]
        covmat *= 1/(2*ls+1)/fsky

        if delta:
            covmat /= delta

        Dl_obs = np.array([Dl_TT, Dl_TE, Dl_EE])
        for i in range(len(ls)):
            if not np.all(np.linalg.eigvals(covmat[:,:,i]) > 0):
                raise Exception("Matrix not positive definite !")
            Dl_obs[:,i] += np.random.multivariate_normal(np.zeros(3), covmat[:,:,i])

        if study == "joint_TT_R_EE":
            Dl_obs[1] /= np.sqrt(Dl_obs[0]*Dl_obs[2])
            covmat[0,1,:] = covmat[1,0,:] = covmat_RTT
            covmat[1,1,:] = covmat_RR
            covmat[1,2,:] = covmat[2,1,:] = covmat_REE

    else:
        raise ValueError("Unknown study '{}'!".format(study))

    # Store simulation informations
    simu = setup["simulation"]
    simu.update({"ls": ls, "Dl": Dl_obs, "covmat": covmat})

def sampling(setup):
    """
    Sample CMB power spectra over cosmo. parameters using `cobaya` using either
    minimization algorithms or MCMC methods.
    """
    from corrcoeff import utils

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    delta = experiment.get("delta", None)
    fsky = experiment["fsky"]
    study = experiment["study"]

    simu = setup["simulation"]
    Dl, cov = simu["Dl"], simu["covmat"]
    if "joint" in study:
        # Invert cov matrix
        inv_cov = np.empty_like(cov)
        for i in range(cov.shape[-1]):
            inv_cov[:,:,i] = np.linalg.inv(cov[:,:, i])

    # Chi2 for CMB spectra sampling
    def chi2(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Dls_theo = _theory.get_Cl(ell_factor=True)
        for s in ["tt", "te", "ee"]:
            Dls_theo[s] = Dls_theo[s][lmin:lmax]
            if delta:
                Dls_theo[s] = utils.bin_array(Dls_theo[s], delta)

        if study == "R":
            R_theo = Dls_theo["te"]/np.sqrt(Dls_theo["tt"]*Dls_theo["ee"])
            chi2 = np.sum((Dl - R_theo)**2/cov)
        else:
            chi2 = np.sum((Dl - Dls_theo[study.lower()])**2/cov)
        return -0.5*chi2

    # Chi2 for joint analysis
    def chi2_joint(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Dls_theo = _theory.get_Cl(ell_factor=True)
        for s in ["tt", "te", "ee"]:
            Dls_theo[s] = Dls_theo[s][lmin:lmax]
            if delta:
                Dls_theo[s] = utils.bin_array(Dls_theo[s], delta)
        Dl_theo = np.array([Dls_theo["tt"], Dls_theo["te"], Dls_theo["ee"]])
        if study == "joint_TT_R_EE":
            Dl_theo[1] /= np.sqrt(Dl_theo[0]*Dl_theo[2])
        delta_Dl = Dl - Dl_theo

        chi2 = 0.0
        for i in range(inv_cov.shape[-1]):
            chi2 += np.dot(delta_Dl[:,i], inv_cov[:,:,i]).dot(delta_Dl[:,i])
        return -0.5*chi2

    # Get cobaya setup
    info = setup["cobaya"]

    # Add likelihood function
    if "joint" in study:
        info["likelihood"] = {"chi2": chi2_joint}
    else:
        info["likelihood"] = {"chi2": chi2}

    from cobaya.run import run
    return run(info)

def store(setup):
    # Store configuration and MINUIT results
    # Remove function pointer and cobaya results (issue with thread)
    if setup.get("cobaya").get("likelihood"):
        del setup["cobaya"]["likelihood"]
    if setup.get("results"):
        results = setup.get("results")
        if results.get("OptimizeResult"):
            del results["OptimizeResult"]["minuit"]
        if results.get("maximum"):
            del results["maximum"]
    import pickle
    pickle.dump({"setup": setup}, open("setup.pkl", "wb"))


# Main function:
def main():
    import argparse
    parser = argparse.ArgumentParser(description="A python program to study correlation between CMB T and E mode")
    parser.add_argument("-y", "--yaml-file", help="Yaml file holding sim/minization setup",
                        default=None, required=True)
    parser.add_argument("--study", help="Set the observable to be studied",
                        choices=["R", "TE", "TT", "EE", "joint_TT_R_EE", "joint_TT_TE_EE"],
                        default=None, required=True)
    parser.add_argument("--seed-simulation", help="Set seed for the simulation random generator",
                        default=None, required=False)
    parser.add_argument("--seed-sampling", help="Set seed for the sampling random generator",
                        default=None, required=False)
    parser.add_argument("--do-minimization", help="Use minimization sampler",
                        default=False, required=False, action="store_true")
    parser.add_argument("--do-mcmc", help="Use MCMC sampler",
                        default=False, required=False, action="store_true")
    parser.add_argument("--use-hessian-covmat", help="Use covariance matrix from minimization",
                        default=False, required=False, action="store_true")
    parser.add_argument("--use-fisher-covmat", help="Use covariance matrix from Fisher calculation",
                        default=False, required=False, action="store_true")
    parser.add_argument("--output-base-dir", help="Set the output base dir where to store results",
                        default=".", required=False)
    parser.add_argument("--systematics-file", help="Set the file name for the systematics",
                        default=None, required=False)
    parser.add_argument("--add-noise", help="Add Simons Observatory noise",
                        default=False, required=False, action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.yaml_file, "r") as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    # Check study
    study = args.study
    setup["experiment"]["study"] = study

    # Check systematics
    if args.systematics_file:
        setup["experiment"]["systematics_file"] = args.systematics_file

    # Check SO noise
    if args.add_noise:
        setup["experiment"]["add_noise"] = args.add_noise

    # Do the simulation
    print("INFO: Doing simulation for '{}'".format(study))
    if args.seed_simulation:
        print("WARNING: Seed for simulation set to {} value".format(args.seed_simulation))
        setup["seed_simulation"] = args.seed_simulation
        np.random.seed(int(args.seed_simulation))
    simulation(setup)

    # Seeding the sampler
    if args.seed_sampling:
        print("WARNING: Seed for sampling set to {} value".format(args.seed_sampling))
        setup["seed_sampling"] = args.seed_sampling
        np.random.seed(int(args.seed_sampling))

    # Do the minimization
    if args.do_minimization:
        setup["cobaya"]["sampler"] = {"minuit": {"ignore_prior": True, "force": True, "ntry_max": 10}}
        setup["cobaya"]["output"] = args.output_base_dir + "/minimize"
        updated_info, results = sampling(setup)
        setup.update({"results": results})

    # Store configuration & results
    store(setup)

    # Do the MCMC
    if args.do_mcmc:
        # Update cobaya setup
        params = setup.get("cobaya").get("params")
        covmat_params = [k for k, v in params.items() if isinstance(v, dict)
                         and "prior" in v.keys() and "proposal" not in v.keys()]
        print("Sampling over", covmat_params, "parameters")
        if args.use_hessian_covmat:
            covmat = results.get("OptimizeResult").get("hess_inv")
            mcmc_dict = {"mcmc": {"covmat": covmat, "covmat_params": covmat_params}}
        elif args.use_fisher_covmat:
            from corrcoeff import utils
            covmat = utils.fisher(setup, covmat_params)
            mcmc_dict = {"mcmc": {"covmat": covmat, "covmat_params": covmat_params}}
        else:
            for p in covmat_params:
                v = params.get(p)
                proposal = (v.get("prior").get("max") - v.get("prior").get("min"))/2
                params[p]["proposal"] = proposal
            mcmc_dict = {"mcmc": None}

        setup["cobaya"]["sampler"] = mcmc_dict
        setup["cobaya"]["output"] = args.output_base_dir + "/mcmc"
        updated_info, results = sampling(setup)

# script:
if __name__ == "__main__":
    main()

# end of corrcoeff.py
