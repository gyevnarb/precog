import precog.utils.log_util as logu
import atexit
import os
import hydra
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(os.path.basename(__file__))


@hydra.main(config_path='conf/esp_evaluate_config.yaml')
def main(cfg):
    output_directory = os.path.realpath(os.getcwd())
    results_directory = cfg.results.dir
    atexit.register(logu.query_purge_directory, output_directory)

    if not hasattr(cfg.goal_detection, "scenario") or cfg.goal_detection.scenario is None:
        scenarios = ["heckstrasse", "bendplatz", "frankenberg"]
    elif isinstance(cfg.goal_detection.scenario, str):
        scenarios = [cfg.goal_detection.scenario]
    else:
        scenarios = cfg.goal_detection.scenario

    log.info(cfg.pretty())

    for scenario in scenarios:
        log.info(f"Summary for scenario {scenario}:")
        results = pd.read_csv(os.path.join(results_directory, f"ind_{scenario}/results.csv"))
        log.info(f"Average number of misses: {results['num_missed'].mean():.3f}+-{results['num_missed'].sem():.3f}")
        log.info(f"Average raw accuracy: {results['raw_accuracy'].mean():.3f}+-{results['raw_accuracy'].sem():.3f}")
        log.info(f"Average adjusted accuracy: {results['adj_accuracy'].mean():.3f}+-{results['adj_accuracy'].sem():.3f}")

        log.info("Average adjusted correct goal probability:")
        goals = range(results["true_goal"].max() + 1)
        for g in goals:
            g_mean = results[results["true_goal"] == g][f"prob_g{g}"].mean()
            g_std = results[results["true_goal"] == g][f"prob_g{g}"].sem()
            log.info(f"\tGoal {g}: {g_mean:.3f}+-{g_std:.3f}")

        goal_probs = results[[f"prob_g{i}" for i in goals]]
        h_goals = goal_probs * np.log2(goal_probs)
        h_goals = -h_goals.sum(axis=1)
        h_uniform = np.log2(len(goals))
        h = h_goals / h_uniform
        log.info(f"Average normalised sample entropy: {h.mean():.3f}+-{h.sem():.3f}")

        log.info("")


if __name__ == '__main__':
    main()
