import functools
import logging
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pdb
import skimage.transform
import tensorflow as tf

import precog.interface as interface

import precog.utils.log_util as logu
import precog.utils.mpl_util as mplu
import precog.utils.tensor_util as tensoru
import precog.ext.nuscenes.nuscenes as nuscenes

from .plot import plot_figure


@logu.log_wrapd()
def plot_sample(sampled_output, expert=None, b=0, figsize=(4, 4), partial_write_np_image_to_tb=None, bev_kwargs={}):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    vis_layer = bev_kwargs["vis_layer"][b]
    vis_scale = bev_kwargs["vis_scale"][b]
    limit = [0, vis_layer.shape[1], vis_layer.shape[0], 0]

    plot_single_sampled_output(sampled_output, batch_index=b, fig=fig, ax=ax,
                               scale=vis_scale, limit=limit)
    ax.imshow(vis_layer, cmap='gray', vmin=0, vmax=255)

    res = plot_figure('sampled_minibatch', fig, partial_write_np_image_to_tb=partial_write_np_image_to_tb)
    plt.close('all')

    return res


def plot_single_sampled_output(sampled_output, batch_index, fig, ax, scale, limit):
    S = sampled_output.rollout.S_world_frame
    S_past = sampled_output.phi.S_past_world_frame
    live_agents = [i for i, traj in enumerate(S_past[batch_index]) if not np.allclose(traj, 0.0)]



    # Plot past.
    plot_joint_trajectory(S_past[batch_index][None], fig=fig, ax=ax, scale=scale, agents=live_agents,
                          marker='d', zorder=2, alpha=0.5, limit=limit)
    # Plot origin.
    plot_joint_trajectory(S_past[batch_index][None][..., -1, :][..., None, :], scale=scale, agents=live_agents,
                          fig=fig, ax=ax, marker='d', zorder=10, alpha=0.5, color='r', limit=limit)

    # Plot future.
    return plot_joint_trajectory(S[batch_index], limit, scale=scale, agents=live_agents,
                                 fig=fig, ax=ax, marker='o', zorder=1, alpha=.4)


def plot_joint_trajectory(joint_traj, limit, scale=1, agents=None, fig=None, ax=None, **kwargs):
    assert (tensoru.rank(joint_traj) == 4)

    if agents is None:
        agents = range(tensoru.size(joint_traj, 1))

    for a in agents:

        if "color" in kwargs:
            color = kwargs.pop("color")
        else:
            color = cm.get_cmap("tab10").colors[a]

        if isinstance(joint_traj, tf.Tensor):
            single_traj = joint_traj[:, a].numpy()
        else:
            single_traj = joint_traj[:, a]

        if scale > 0:
            single_traj *= scale
            single_traj[..., -1] *= -1

        fig, ax = plot_trajectory(single_traj, color=color, fig=fig, ax=ax, axis=limit, **kwargs)
        assert (fig is not None)
    return fig, ax


def plot_trajectory(traj, fig=None, ax=None, alpha=None, zorder=1, marker='o', markeredgewidth=None, markersize=None,
                    markerfacecolor='none', markeredgecolor=None, linewidth=1, color=None, label=None, axis=None):
    assert fig is not None
    ax.plot(*traj.T,
            marker=marker,
            linewidth=linewidth,
            color=color,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            alpha=alpha,
            zorder=zorder,
            label=label)

    if axis is not None:
        ax.axis(axis)
    return fig, ax
