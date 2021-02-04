import atexit
import logging
import hydra
import functools
import pandas as pd
import numpy as np
import os
import scipy.stats
import skimage.io

import precog.utils.log_util as logu
import precog.utils.tfutil as tfutil
import precog.interface as interface
import precog.plotting.plot as plot
import precog.plotting.plot_ind as plot_ind
import precog.dataset.ind_util as ind_util

log = logging.getLogger(os.path.basename(__file__))


def remove_future_tensors(cfg, inference, minibatch):
    if not cfg.main.compute_metrics:
        for t in inference.training_input.experts.tensors:
            try:
                del minibatch[t]
            except KeyError:
                pass


@hydra.main(config_path='conf/esp_infer_config.yaml')
def main(cfg):
    output_directory = os.path.realpath(os.getcwd())
    images_directory = os.path.join(output_directory, 'images')
    os.mkdir(images_directory)
    os.mkdir(os.path.join(images_directory, 'rollout'))
    log.info("\n\nConfig:\n===\n{}".format(cfg.pretty()))

    atexit.register(logu.query_purge_directory, output_directory)

    # Instantiate the session.
    sess = tfutil.create_session(
        allow_growth=cfg.hardware.allow_growth,
        per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)

    # Load the model and the tensor collections.
    log.info("Loading the model...")
    ckpt, graph, tensor_collections = tfutil.load_annotated_model(cfg.model.directory, sess)
    inference = interface.ESPInference(tensor_collections)
    sample_metrics = tfutil.get_collection_dict(tensor_collections['sample_metric'])

    if cfg.main.compute_metrics:
        infer_metrics = tfutil.get_collection_dict(tensor_collections['infer_metric'])
        metrics = {**infer_metrics, **sample_metrics}
        all_metrics = {_: [] for _ in metrics.keys()}

    # Instantiate the dataset.
    cfg.dataset.params.T = inference.metadata.T
    cfg.dataset.params.B = inference.metadata.B
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)

    log.info("Beginning evaluation. Model: {}".format(ckpt))
    count = 0
    rollout_count = 0
    start_batch_number = 0
    while True:
        minibatch, metadata = dataset.get_minibatch(split=cfg.split, input_singleton=inference.training_input,
                                                    is_training=False, return_metadata=True)
        remove_future_tensors(cfg, inference, minibatch)
        if minibatch is None: break

        sessrun = functools.partial(sess.run, feed_dict=minibatch)
        try:
            # Run sampling and convert to numpy.
            sampled_output_np = inference.sampled_output.to_numpy(sessrun)

            if cfg.main.compute_metrics:
                # Get experts in numpy version.
                experts_np = inference.training_input.experts.to_numpy(sessrun)
                # Compute and store metrics.
                metrics_results = dict(zip(metrics.keys(), sessrun(list(metrics.values()))))
                for k, val in metrics_results.items():
                    all_metrics[k].append(val)
            else:
                experts_np = None
        except ValueError as v:
            print("Got value error: '{}'\n Are you sure the provided dataset ('{}') "
                  "is compatible with the model?".format(v, cfg.dataset))
            raise v

        if cfg.main.compute_goals:
            log.info(f"Running goal recognition for {cfg.goal_detection.try_count} iterations:")
            S_past = sampled_output_np.phi.S_past_world_frame  # (B, A, Tp, D)
            S = sampled_output_np.rollout.S_world_frame
            K = sampled_output_np.rollout.S_world_frame.shape[1]
            trajectories = np.repeat(S_past.copy()[:, np.newaxis, ...], K, axis=1)  # (B, K, A, try_count * Tp, D)
            trajectories = np.append(trajectories, S, axis=3)

            for i in range(cfg.goal_detection.try_count):
                goal_completion, dones = ind_util.InDGoalDetector.update_precog_completion(
                    cfg, trajectories, S_past, scale=metadata["vis_scale"][0])
                log.info(f"Rollout {i} - Goal reached for agents "
                         f"{[i for i, b in enumerate(goal_completion) if len(b) == 0]}")
                if all(dones):
                    log.info("All agents' goal reached")
                    break

                synthetic_data = ind_util.InDMultiagentDatum.from_precog_predictions(
                    cfg, S, S_past, sampled_output_np.phi.overhead_features, metadata)

                S = ind_util.rollout_future(synthetic_data, cfg, sess, inference, dataset)
                trajectories = np.append(trajectories, S, axis=3)

            goal_completion, dones = ind_util.InDGoalDetector.update_precog_completion(
                cfg, trajectories, S_past, scale=metadata["vis_scale"][0])

            if cfg.goal_detection.plot_rollout:
                for b in range(cfg.dataset.params.B):
                    vis_layer = metadata["vis_layer"][b]
                    limit = [0, vis_layer.shape[1], vis_layer.shape[0], 0]
                    im = plot_ind.plot_rollout_trajectories(b, trajectories, S_past,
                                                            figsize=cfg.images.figsize,
                                                            background=vis_layer, limit=limit)
                    skimage.io.imsave(f'{images_directory}/rollout/esp_rollout_{rollout_count:05d}.{cfg.images.ext}',
                                      im[0, ..., :3])
                    log.info("Rollout plotted.")
                    rollout_count += 1

            results = ind_util.InDGoalDetector.predict_proba(goal_completion,
                                                             metadata["true_goals"],
                                                             trajectories, S_past, cfg,
                                                             metadata["vis_scale"][0],
                                                             start_batch_number)
            results.to_csv(os.path.join(output_directory, "results.csv"),
                           mode="a", index=False, header=(start_batch_number == 0))

        if cfg.main.plot:
            log.info("Plotting...")
            for b in range(sampled_output_np.phi.S_past_world_frame.shape[0]):
                if cfg.main.ind:
                    S_past = sampled_output_np.phi.S_past_world_frame
                    S = sampled_output_np.rollout.S_world_frame
                    im = plot_ind.plot_sample(S, S_past,
                                              b=b,
                                              partial_write_np_image_to_tb=lambda x: x,
                                              figsize=cfg.images.figsize,
                                              bev_kwargs=metadata)
                else:
                    im = plot.plot_sample(sampled_output_np,
                                          experts_np,
                                          b=b,
                                          partial_write_np_image_to_tb=lambda x: x,
                                          figsize=cfg.images.figsize,
                                          bev_kwargs=cfg.plotting.bev_kwargs)
                skimage.io.imsave('{}/esp_samples_{:05d}.{}'.format(images_directory, count, cfg.images.ext),
                                  im[0, ..., :3])
                log.info("Plotted.")
                count += 1

        if cfg.main.compute_metrics:
            for k, vals in all_metrics.items():
                log.info("Mean,sem '{}'={:.3f} +- {:.3f}".format(k, np.mean(vals), scipy.stats.sem(vals, axis=None)))

        start_batch_number += cfg.dataset.params.B

    if cfg.main.compute_metrics:
        log.info("Final metrics\n=====\n")
        for k, vals in all_metrics.items():
            log.info("Mean,sem '{}'={:.3f} +- {:.3f}".format(k, np.mean(vals), scipy.stats.sem(vals, axis=None)))


if __name__ == '__main__': main()
