import argparse
import logging
import os
import sys
from pathlib import Path
from pytorch_lightning.utilities import AttributeDict

import numpy as np
import torch

sys.path.append(".")

from reid_config import cfg
from train_mctl_model import CTLModel
from reid_utils.reid_metric import get_dist_func
from inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    make_inference_data_loader,
    run_inference,
)

### Prepare logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create embeddings for images that will serve as the database (gallery)"
    )
    parser.add_argument(
        "--config_file", default="D://PyCharm/Order_Rreceiving/1000/ReID/centroid/configs/256_resnet50.yml",
        help="path to config file", type=str
    )
    parser.add_argument(
        "--model_path", default="../../weights/msmt17_best.ckpt",
        help="the trained model", type=str
    )
    parser.add_argument(
        "--images-in-subfolders",
        help="if images are stored in the subfloders use this flag. If images are directly under DATASETS.ROOT_DIR path do not use it.",
        action="store_true",
    )
    parser.add_argument(
        "--print_freq",
        help="number of batches the logging message is printed",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--query_dir",
        default="../../centroid/inference_datasets/msmt17/query/",
        help="query data paths in the dataset",
        type=str,
    )
    parser.add_argument(
        "--gallery_data",
        default="../../centroid/inference/result/msmt17",
        help="path to root where previously prepared embeddings and paths were saved",
        type=str,
    )
    parser.add_argument(
        "--normalize_features",
        help="whether to normalize the gallery and query embeddings",
        action="store_true",
    )
    parser.add_argument(
        "--topk",
        help="number of top k similar ids to return per query. If set to 0 all ids per query will be returned",
        type=int,
        default=10,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output",
        help="the path saved by the result",
        default="result/msmt17",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    ### Data preparation
    if args.images_in_subfolders:
        dataset_type = ImageFolderWithPaths
    else:
        dataset_type = ImageDataset
    log.info(f"Preparing data using {type(dataset_type)} dataset class")
    val_loader = make_inference_data_loader(cfg, args.query_dir, dataset_type)
    use_cuda = torch.cuda.is_available()
    ### Build model
    model = CTLModel.load_from_checkpoint(args.model_path)

    ### Inference
    log.info("Running inference")
    embeddings, paths = run_inference(
        model, val_loader, cfg, print_freq=args.print_freq, use_cuda=use_cuda
    )

    ### Load gallery data
    LOAD_PATH = Path(args.gallery_data)
    embeddings_gallery = torch.from_numpy(
        np.load(LOAD_PATH / "embeddings.npy", allow_pickle=True)
    )
    paths_gallery = np.load(LOAD_PATH / "paths.npy", allow_pickle=True)

    if args.normalize_features:
        embeddings_gallery = torch.nn.functional.normalize(
            embeddings_gallery, dim=1, p=2
        )
        embeddings = torch.nn.functional.normalize(
            torch.from_numpy(embeddings), dim=1, p=2
        )
    else:
        embeddings = torch.from_numpy(embeddings)

    # Use GPU if available
    # device = torch.device("cuda") if cfg.GPU_IDS else torch.device("cpu")
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    embeddings_gallery = embeddings_gallery.to(device)
    embeddings = embeddings.to(device)

    ### Calculate similarity
    log.info("Calculating distance and getting the most similar ids per query")
    dist_func = get_dist_func(cfg.SOLVER.DISTANCE_FUNC)
    distmat = dist_func(x=embeddings, y=embeddings_gallery).cpu().numpy()
    indices = np.argsort(distmat, axis=1)

    ### Constrain the results to only topk most similar ids
    indices = indices[:, : args.topk] if args.topk else indices

    out = {
        query_path: {
            "indices": indices[q_num, :],
            "paths": paths_gallery[indices[q_num, :]],
            "distances": distmat[q_num, indices[q_num, :]],
        }
        for q_num, query_path in enumerate(paths)
    }

    # 将匹配好的图片存放txt文件
    img_similer_file = open(os.path.join(Path(args.output), 'similer_imgs.txt'), 'w', encoding='utf-8')
    similar = AttributeDict()
    for q_num, query_path in enumerate(paths):
        similar[query_path] = paths_gallery[indices[q_num, :]]

    for k in similar:
        result = str(k) + ":" + ",".join([str(x) for x in similar[k]])
        img_similer_file.write(result + "\n")

    ### Save
    SAVE_DIR = Path(args.output)
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    log.info(f"Saving results to {str(SAVE_DIR)}")
    np.save(SAVE_DIR / "results.npy", out)
    np.save(SAVE_DIR / "query_embeddings.npy", embeddings)
    np.save(SAVE_DIR / "query_paths.npy", paths)
