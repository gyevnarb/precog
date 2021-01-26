import atexit
import logging
import hydra
import functools
import dill
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
    assert (cfg.main.plot or cfg.main.compute_metrics)
    output_directory = os.path.realpath(os.getcwd())
    images_directory = os.path.join(output_directory, 'images')
    os.mkdir(images_directory)
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
    if cfg.main.compute_goals:
        goal_metrics = {}

    # Instantiate the dataset.
    cfg.dataset.params.T = inference.metadata.T
    cfg.dataset.params.B = inference.metadata.B
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)

    log.info("Beginning evaluation. Model: {}".format(ckpt))
    count = 0
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

        if False and cfg.main.plot:
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

        if cfg.main.compute_goals:
            log.info(f"Running goal recognition for {cfg.goal_detection.try_count} iterations:")
            S_past_ = sampled_output_np.phi.S_past_world_frame  # (B, A, Tp, D)
            S = sampled_output_np.rollout.S_world_frame
            K = sampled_output_np.rollout.S_world_frame.shape[1]
            goal_completion = [[] for _ in range(cfg.dataset.params.B)]
            trajectories = np.repeat(S_past_.copy()[:, np.newaxis, ...], K, axis=1)

            for i in range(cfg.goal_detection.try_count):
                done = ind_util.GoalDetector.update_precog_completion(goal_completion, cfg, S, S_past_)
                if done: break

                synthetic_data = ind_util.InDMultiagentDatum.from_precog_predictions(
                    cfg, S, S_past_, sampled_output_np.phi.overhead_features, metadata)
                S = []
                S_past = []
                # Iterate over each sample in every minibatch and feed predictions back into the model
                for s in range(K):

                    minibatch = dataset.get_minibatch(split=cfg.split, input_singleton=inference.training_input,
                                                      is_training=False, return_metadata=False,
                                                      data_feed=synthetic_data[s])
                    remove_future_tensors(cfg, inference, minibatch)
                    if minibatch is None: break

                    sessrun = functools.partial(sess.run, feed_dict=minibatch)
                    sample = inference.sampled_output.to_numpy(sessrun)
                    sample_future = sample.rollout.S_world_frame[:, [0], ...]  # Only keep the first sampling
                    S.append(sample_future)
                    S_past.append(sample.phi.S_past_world_frame)
                S = np.hstack(S)
                S_past = np.stack(S_past, axis=1)
                trajectories = np.append(trajectories, S_past, axis=3)

        if cfg.main.compute_metrics:
            for k, vals in all_metrics.items():
                log.info("Mean,sem '{}'={:.3f} +- {:.3f}".format(k, np.mean(vals), scipy.stats.sem(vals, axis=None)))

    if cfg.main.compute_metrics:
        log.info("Final metrics\n=====\n")
        for k, vals in all_metrics.items():
            log.info("Mean,sem '{}'={:.3f} +- {:.3f}".format(k, np.mean(vals), scipy.stats.sem(vals, axis=None)))


if __name__ == '__main__': main()
