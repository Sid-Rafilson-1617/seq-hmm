import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from custom_hmm import PoissonHMM, initialization, make_transition_update_mask
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--save-dir", required=True)

    p.add_argument("--r", type=float, default=0.8)

    p.add_argument("--nStates", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.1)

    p.add_argument("--restart", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)

    p.add_argument("--Niters", type=int, default=100)
    p.add_argument("--eps", type=float, default=1e-12)

    p.add_argument("--plot", type=bool, default=False)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()



def main(args):

    # set the random seed for reproducibility
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    # define the current save directory for this restart and transition matrix size
    current_save_dir = os.path.join(args.save_dir, f"N_{args.nStates}", f"restart_{args.restart}")
    os.makedirs(current_save_dir, exist_ok=True)

    # load the data
    data = np.load(args.data)


    # get the transition update mask to specify which transitions to update during fitting
    transition_update_mask = make_transition_update_mask(np.zeros((args.nStates, args.nStates)))

    # get random initial parameters for the HMM
    initial_A, initial_B, initial_pi = initialization(
        args.nStates,data['observations'],
        transition_update_mask)

    # define the model and set the observations
    model = PoissonHMM(
        A = initial_A,
        B = initial_B,
        pi = initial_pi,
        eps = args.eps
    )

    model.set_observations(data['observations'])

    # get the transition update mask to specify which transitions to update during fitting
    transition_update_mask = make_transition_update_mask(initial_A)

    # run the custom baum-welch algorithm with cloned states
    training_likelihoods = model.fit_em(
        Niters=args.Niters,
        use_cloned_emissions=True,
        transition_update_mask=transition_update_mask,
        save_dir=None
    )



    # plot the EM convergence
    if args.plot:
        plt.figure()
        plt.plot(np.asarray(training_likelihoods))
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
    np.savez(
        os.path.join(current_save_dir, "fitted_model.npz"),
        A = model.A,
        B = model.B,
        pi= model.pi,
        training_likelihoods = training_likelihoods,
        nStates = args.nStates,
        restart = args.restart,
        seed = args.seed
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--nStates", type=int, required=True)
    parser.add_argument("--restart", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--Niters", type=int, required=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()
    main(args)
