import numpy as np
from custom_hmm import PoissonHMM
from scipy import stats
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-params", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--neuron-index", type=int, required=True)

    return p.parse_args()




def main(args):


    # define the current save directory for this hold-out neuron
    current_save_dir = os.path.join(args.save_dir, f"neuron_{args.neuron_index}")
    os.makedirs(current_save_dir, exist_ok=True)

    # load the data
    data = np.load(args.data, allow_pickle=True)
    observations = data["observations"]
    time_bins = data["time_bins"]

    # load the model params
    A, B, pi = np.load(args.model_params, allow_pickle=True).values()

    # initialize the HMM
    train_observations = np.delete(observations, args.neuron_index, axis=1)
    test_observations = observations[:, args.neuron_index]

    model = PoissonHMM(A, np.delete(B, args.neuron_index, axis=1), pi)

    model.set_observations(train_observations)

    # run the viterbi algorithm
    states = model.viterbi()

    # compute the likelihood of the test observations given the inferred states
    log_likelihoods = np.array(stats.poisson.logpmf(test_observations, B[:, args.neuron_index][states]))

    # save the log likelihoods
    np.save(os.path.join(current_save_dir, "log_likelihoods.npy"), log_likelihoods)

    # also save the inferred states
    np.save(os.path.join(current_save_dir, "states.npy"), states)


if __name__ == "__main__":
    args = parse_args()
    main(args)







