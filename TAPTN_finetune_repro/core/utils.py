import os
import numpy as np
import time
import datetime
import pytz


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


def _default_pretrained_root():
    if os.environ.get('TAPTN_PRETRAINED'):
        return os.environ['TAPTN_PRETRAINED']
    if os.environ.get('TAPTN_ASSETS'):
        return os.path.join(os.environ['TAPTN_ASSETS'], 'pretrained')
    return 'pretrained'


def lm_pretrained_path(model_name, root=None):
    """Local HuggingFace weights: {root}/{model_name}."""
    if root is None:
        root = _default_pretrained_root()
    return os.path.join(root, model_name)


def lm_model_root(cfg):
    return getattr(cfg.lm.model, 'root', None) or _default_pretrained_root()


def lm_artifact_root(cfg):
    """可选：prt_lm/{artifact_root}/... 与 output/{artifact_root}/... 隔离跨模型产物。"""
    ar = getattr(cfg.lm.model, 'artifact_root', '') or ''
    return ar.strip('/')


def lm_rel_prefix(cfg, dataset_name, use_gpt_str=''):
    base = f'{dataset_name}{use_gpt_str}'
    ar = lm_artifact_root(cfg)
    return f'{ar}/{base}' if ar else base


def lm_ckpt_stem(cfg, dataset_name, use_gpt_str, seed):
    model_name = cfg.lm.model.name
    rel = lm_rel_prefix(cfg, dataset_name, use_gpt_str)
    return f'prt_lm/{rel}/{model_name}-seed{seed}'


def lm_output_stem(cfg, dataset_name, use_gpt_str, seed):
    model_name = cfg.lm.model.name
    rel = lm_rel_prefix(cfg, dataset_name, use_gpt_str)
    return f'output/{rel}/{model_name}-seed{seed}'


def lm_emb_path(cfg, dataset_name, feature_type, seed):
    """feature_type: TA→P1, E→P2(sem), EN→P3(nosem)。"""
    suffix_map = {'TA': '', 'E': '3', 'EN': '3nosem'}
    suffix = suffix_map.get(feature_type, '')
    return f'{lm_ckpt_stem(cfg, dataset_name, suffix, seed)}.emb'


def infer_emb_dim(emb_path, num_nodes, feat_shrink=''):
    if feat_shrink:
        return int(feat_shrink)
    if not os.path.isfile(emb_path) or num_nodes <= 0:
        return 768
    fs = os.path.getsize(emb_path)
    dim = fs // (num_nodes * 2)   # float16
    return dim if dim > 0 else 768


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper
