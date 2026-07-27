import numpy as np
from custom_hmm import PoissonHMM
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-params", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--niters", type=int, default=20)  # number of EM iterations
    p.add_argument("--save-dir", required=True)
    p.add_argument("--restart", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)

    return p.parse_args()


def main(args):


    # define the current save directory for this hold-out neuron
    save_dir = os.path.join(args.save_dir, f"restart_{args.restart}")
    os.makedirs(save_dir, exist_ok=True)

    # load the data
    data = np.load(args.data, allow_pickle=True)
    observations = data["observations"]
    time_bins = data["time_bins"]

    with np.load(args.model_params, allow_pickle=True) as params:
        A = params["A"].copy()
        B = params["B"].copy()
        pi = params["pi"].copy()

    # Preserve the model's transition topology
    transition_update_mask = A > 0

    rng = np.random.default_rng(args.seed)

    # Positive multiplicative perturbations
    A *= np.exp(0.01 * rng.standard_normal(A.shape))
    B *= np.exp(0.01 * rng.standard_normal(B.shape))
    pi *= np.exp(0.01 * rng.standard_normal(pi.shape))

    # Keep forbidden transitions at exactly zero
    A[~transition_update_mask] = 0.0

    # Guard against numerical underflow
    A[transition_update_mask] = np.clip(
        A[transition_update_mask], 1e-12, None
    )
    B = np.clip(B, 1e-12, None)
    pi = np.clip(pi, 1e-12, None)

    # Normalize only probability parameters
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    model = PoissonHMM(A, B, pi)
    

    model.set_observations(observations)

    # fit with em
    llh = model.fit_em(args.niters, use_cloned_emissions = True, transition_update_mask=transition_update_mask)

    # save the updated model parameters and log-likelihoods
    np.savez(os.path.join(save_dir, "updated_model.npz"), A=model.A, B=model.B, pi=model.pi)
    np.save(os.path.join(save_dir, "log_likelihoods.npy"), llh)

if __name__ == "__main__":
    args = parse_args()
    main(args)
