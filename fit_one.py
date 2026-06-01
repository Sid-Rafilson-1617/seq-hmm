import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from custom_hmm import PoissonHMM, initialization, make_transition_update_mask, cv_split
import argparse
import os


def main(args):

    # define the current save directory for this fold and transition matrix size
    current_save_dir = os.path.join(args.save_dir, f"fold_{args.fold}", f"N_{args.nStates}")

    # load the data (expects a .npz file with an array called 'observations')
    data = np.load(args.data)

    # get the length of the observation sequence
    T = data['observations'].shape[0]

    # get the train and test indices for cross-validation
    train_idx, test_idx = cv_split(K = args.K, fold = args.fold, r = args.r, T = T, verbose = args.verbose)

    # get random initial parameters for the HMM
    initial_A, initial_B, initial_pi = initialization(data['observations'][train_idx], args.nStates)

    # define the model and set the observations
    model = PoissonHMM(A = initial_A, B = initial_B, pi = initial_pi, eps = args.eps)
    model.set_observations(data['observations'][train_idx])

    # get the transition update mask to specify which transitions to update during fitting
    transition_update_mask = make_transition_update_mask(initial_A)

    # run the custom baum-welch algorithm with cloned states
    lls = model.fit_em(Niters=args.Niters, use_cloned_emissions=True, transition_update_mask=transition_update_mask, save_dir=current_save_dir)

    # plot the EM convergence
    if args.plot:
        plt.figure()
        plt.plot(np.asarray(lls))
        plt.title(f"EM Convergence")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log-Likelihood")
        plt.tight_layout()
        plt.savefig(os.path.join(current_save_dir, f"em_convergence.png"))
        plt.close()

        # plot the transition matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array(model.A), cmap="Blues", cbar_kws={"label": "Transition Probability (%)"}, vmin=0, vmax=1)
        plt.title("Learned Transition Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.tight_layout()
        plt.savefig(os.path.join(current_save_dir, f"learned_transition_matrix.png"))
        plt.close()


    # save the A and B matrices and the log likelihoods to the current save directory
    np.savez(os.path.join(current_save_dir, f"hmm_fit.npz"), A=model.A, B=model.B, pi=model.pi, log_likelihoods=lls)

