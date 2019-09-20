# Global
import numpy as np

def simulation(setup):
    """
    Simulate CMB power spectrum given a set of cosmological parameters and noise level.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    fsky = experiment["fsky"]

    from corrcoeff import utils
    Cls = utils.get_theory_cls(setup, lmax)
    ls = np.arange(lmin, lmax)
    Cl_TT = Cls["tt"][lmin:lmax]
    Cl_TE = Cls["te"][lmin:lmax]
    Cl_EE = Cls["ee"][lmin:lmax]

    if experiment.get("systematics_file"):
        syst = np.loadtxt(experiment["systematics_file"])
        syst = syst[:,-1][lmin:lmax]
        Cl_TE *= syst
        Cl_TT *= syst
        Cl_EE *= syst

    if experiment.get("add_noise"):
        # Get SO noise
        N_TT, N_EE = utils.get_noise(experiment)
        N_TT, N_EE = 1/np.sum(1/N_TT, axis=0), 1/np.sum(1/N_EE, axis=0)
    else:
        N_TT = 0.0
        N_EE = 0.0

    R = Cl_TE/np.sqrt(Cl_TT*Cl_EE)

    covmat_RR   = R**4 - 2*R**2 + 1 + N_TT/Cl_TT + N_EE/Cl_EE + (N_TT*N_EE)/(Cl_TT*Cl_EE) \
        + R**2*(0.5*(N_TT/Cl_TT - 1)**2 + 0.5*(N_EE/Cl_EE - 1)**2 - 1)
    covmat_TTTT = 2*(Cl_TT+N_TT)**2
    covmat_TETE = (Cl_TT+N_TT)*(Cl_EE+N_EE) + Cl_TE**2
    covmat_EEEE = 2*(Cl_EE+N_EE)**2

    study = experiment["study"]
    if study == "R":
        # Compute TE correlation factor
        covmat = 1/(2*ls+1)/fsky*covmat_RR
        Cl_obs = R + np.sqrt(covmat)*np.random.randn(len(ls))
    elif study == "TT":
        covmat = 1/(2*ls+1)/fsky*covmat_TTTT
        Cl_obs = Cl_TT + np.sqrt(covmat)*np.random.randn(len(ls))
    elif study == "TE":
        covmat = 1/(2*ls+1)/fsky*covmat_TETE
        Cl_obs = Cl_TE + np.sqrt(covmat)*np.random.randn(len(ls))
    elif study == "EE":
        covmat = 1/(2*ls+1)/fsky*covmat_EEEE
        Cl_obs = Cl_EE + np.sqrt(covmat)*np.random.randn(len(ls))
    elif "joint" in study:
        covmat_TTEE = 2*Cl_TE**2
        covmat_TTTE = 2*(Cl_TT+N_TT)*Cl_TE
        covmat_TEEE = 2*(Cl_EE+N_EE)*Cl_TE
        covmat_REE  = R*(covmat_TEEE/Cl_TE - 0.5*covmat_EEEE/Cl_EE - 0.5*covmat_TTEE/Cl_TT)
        covmat_RTT  = R*(covmat_TTTE/Cl_TE - 0.5*covmat_TTEE/Cl_EE - 0.5*covmat_TTTT/Cl_TT)

        covmat = np.empty((3, 3, len(ls)))
        covmat[0,0,:] = covmat_TTTT
        covmat[0,1,:] = covmat_TTTE
        covmat[0,2,:] = covmat_TTEE
        covmat[1,1,:] = covmat_TETE
        covmat[1,2,:] = covmat_TEEE
        covmat[2,2,:] = covmat_EEEE

        Cl_obs = np.array([Cl_TT, Cl_TE, Cl_EE])
        if study == "joint_TT_R_EE":
            Cl_obs[1] = R
            covmat[0,1,:] = covmat_RTT
            covmat[1,1,:] = covmat_RR
            covmat[1,2,:] = covmat_REE

        covmat[1,0,:] = covmat[0,1,:]
        covmat[2,0,:] = covmat[0,2,:]
        covmat[2,1,:] = covmat[1,2,:]
        covmat *= 1/(2*ls+1)/fsky

        for i in range(len(ls)):
            if not np.all(np.linalg.eigvals(covmat[:,:,i]) > 0):
                raise Exception("Matrix not positive definite !")
            Cl_obs[:,i] += np.random.multivariate_normal(np.zeros(3), covmat[:,:,i])

    else:
        raise ValueError("Unknown study '{}'!".format(study))

    # Store simulation informations
    simu = setup["simulation"]
    simu.update({"Cl": Cl_obs, "covmat": covmat})

def sampling(setup):
    """
    Sample CMB power spectra over cosmo. parameters using `cobaya` using either
    minimization algorithms or MCMC methods.
    """

    # Get experiment setup
    experiment = setup["experiment"]
    lmin, lmax = experiment["lmin"], experiment["lmax"]
    fsky = experiment["fsky"]
    study = experiment["study"]

    simu = setup["simulation"]
    Cl, cov = simu["Cl"], simu["covmat"]
    if "joint" in study:
        # Invert cov matrix
        inv_cov = np.empty_like(cov)
        for i in range(cov.shape[-1]):
            inv_cov[:,:,i] = np.linalg.inv(cov[:,:, i])

    # Chi2 for CMB spectra sampling
    def chi2(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Cls_theo = _theory.get_cl(ell_factor=False)
        for s in ["tt", "te", "ee"]:
            Cls_theo[s] = Cls_theo[s][lmin:lmax]
        if study == "R":
            R_theo = Cls_theo["te"]/np.sqrt(Cls_theo["tt"]*Cls_theo["ee"])
            chi2 = np.sum((Cl - R_theo)**2/cov)
        else:
            chi2 = np.sum((Cl - Cls_theo[study.lower()])**2/cov)
        return -0.5*chi2

    # Chi2 for joint analysis
    def chi2_joint(_theory={"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}):
        Cls_theo = _theory.get_cl(ell_factor=False)
        for s in ["tt", "te", "ee"]:
            Cls_theo[s] = Cls_theo[s][lmin:lmax]
        Cl_theo = np.array([Cls_theo["tt"], Cls_theo["te"], Cls_theo["ee"]])
        if study == "joint_TT_R_EE":
            Cl_theo[1] /= np.sqrt(Cl_theo[0]*Cl_theo[2])
        delta = Cl - Cl_theo

        chi2 = 0.0
        for i in range(inv_cov.shape[-1]):
            chi2 += np.dot(delta[:,i], inv_cov[:,:,i]).dot(delta[:,i])
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

    # Store configuration & results
    store(setup)

# script:
if __name__ == "__main__":
    main()

# end of corrcoeff.py
