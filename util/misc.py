import os
import sys
import logging
import torch
# import torch.distributed as dist
import hfai.distributed as dist


def init_dist(local_rank):
    # init dist
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = os.getenv("MASTER_PORT", 1024)
    hosts = int(os.getenv("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.getenv("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)
    dist.barrier()

    return dist.get_rank(), dist.get_world_size()


def setup_logger(output_dir, basename='log', distributed_rank=0):
    # ---- set up logger ----
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    # each experiment's output is in the dir named after the time when it starts to run
    # log_dir_name = log_prefix + '-[{}]'.format((datetime.datetime.now()).strftime('%m%d%H%M%S'))

    os.makedirs(output_dir, exist_ok=True)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", '%m%d%H%M%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if basename[-4:] == '.txt':
        basename = basename[:-4]
    # basename = basename+str(distributed_rank)
    txt_name = f'{basename}-1.txt'
    for i in range(2, 100000):
        if os.path.exists(os.path.join(output_dir, txt_name)):
            cur_basename = f'{basename}-{i}'
            txt_name = f'{cur_basename}.txt'
        else:
            break
    fh = logging.FileHandler(os.path.join(output_dir, txt_name), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_info(log_str):
    logger = logging.getLogger()
    assert len(logger.handlers), "No logger handlers."
    logger.info(log_str)


