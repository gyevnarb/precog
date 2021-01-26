import numpy as np
import tensorflow as tf

import precog.interface as interface


def nuscenes_dill_metadata_producer(data):
    items = interface.MetadataList()
    sample_tokens = np.asarray([_.metadata['scene_token'] for _ in data])
    items.append(interface.MetadataItem(name='sample_token', array=sample_tokens, dtype=tf.string))
    # scene_tokens = np.asarray([_.metadata['real_scene_token'] for _ in data])
    # items.append(interface.MetadataItem(name='scene_token', array=scene_tokens, dtype=tf.string)) 
    return items


def nuscenes_mini_dill_metadata_producer(data):
    items = interface.MetadataList()
    sample_tokens = np.asarray([_.metadata['sample_token'] for _ in data])
    items.append(interface.MetadataItem(name='sample_token', array=sample_tokens, dtype=tf.string))
    # scene_tokens = np.asarray([_.metadata['real_scene_token'] for _ in data])
    # items.append(interface.MetadataItem(name='scene_token', array=scene_tokens, dtype=tf.string)) 
    return items


def carla_town01_A5_T20_metadata_producer(data):
    return interface.MetadataList()


def carla_town01_A1_T20_metadata_producer(data):
    return interface.MetadataList()


def carla_town01_A1_T20_lightstate_metadata_producer(data):
    return interface.MetadataList()


def ind_heckstrasse_metadata_producer(data):
    items = interface.MetadataList()

    vis_layer = np.asarray([data[i].metadata["vis_layer"] for i in range(len(data))])
    items.append(interface.MetadataItem(name='vis_layer', array=vis_layer, dtype=np.int32))

    vis_scale = np.asarray([data[i].metadata["vis_scale"] for i in range((len(data)))])
    items.append(interface.MetadataItem(name="vis_scale", array=vis_scale, dtype=np.float64))

    agent_dims = [data[i].metadata["agent_dims"] for i in range(len(data))]
    items.append(interface.MetadataItem(name="agent_dims", array=agent_dims, dtype=None))

    return items


PRODUCERS = {
    "trimodal_dataset": lambda *args, **kwargs: interface.MetadataList(),
    'ind_heckstrasse_dill': ind_heckstrasse_metadata_producer,
    'ind_bendplatz_dill': lambda *args, **kwargs: interface.MetadataList(),
    'ind_frankenberg_dill': lambda *args, **kwargs: interface.MetadataList(),
    'nuscenes_shuffle_A5_dill': nuscenes_dill_metadata_producer,
    'nuscenes_mini_dill': nuscenes_mini_dill_metadata_producer,
    'carla_town01_A5_T20_json': carla_town01_A5_T20_metadata_producer,
    'carla_town01_A1_T20_json': carla_town01_A1_T20_metadata_producer,
    'carla_town01_A1_T20_lightstate_json': carla_town01_A1_T20_lightstate_metadata_producer,
    'carla_town01_A1_T20_lightstate_streamingloader_json': carla_town01_A1_T20_lightstate_metadata_producer,
    'carla_town01_A1_T20_v1_json': carla_town01_A1_T20_lightstate_metadata_producer,
    'carla_town01_A1_T20_v2_json': carla_town01_A1_T20_lightstate_metadata_producer
}
