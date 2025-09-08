import os
import numpy as np
from typing import Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import logging

from config_and_utils import SignalConfig, set_seed, configure_logging
from generators import synthesize_modulation
from channel import pass_through_channel


def complex_to_2chan(x: np.ndarray) -> np.ndarray:
    return np.stack([np.real(x), np.imag(x)], axis=-1).astype(np.float32)


def build_dataset(
    cfg: SignalConfig,
    examples_per_class_per_snr: int = 500,
    use_multipath: bool = True,
    save_dir: str = "data",
    tag: str = "v1",
    seed: int = 42,
    logger: logging.Logger = None
) -> Dict[str, str]:
    if logger is None:
        logger = logging.getLogger("modulation")
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    classes = list(cfg.classes)
    snrs = list(cfg.snr_db_grid)

    X_list = []
    y_list = []
    snr_list = []

    logger.info("Generating raw examples...")
    for mod_idx, mod in enumerate(tqdm(classes)):
        for snr_db in snrs:
            for _ in range(examples_per_class_per_snr):
                base = synthesize_modulation(mod, cfg)
                ysig = pass_through_channel(base, cfg, snr_db=snr_db, use_multipath=use_multipath)
                X_list.append(complex_to_2chan(ysig))
                y_list.append(mod_idx)
                snr_list.append(snr_db)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    snr_arr = np.array(snr_list, dtype=np.float32)

    snr_to_idx = {s: i for i, s in enumerate(snrs)}
    strat = np.array([f"{yi}_{snr_to_idx[float(si)]}" for yi, si in zip(y, snr_arr)])

    X_train, X_tmp, y_train, y_tmp, snr_train, snr_tmp, strat_train, strat_tmp = train_test_split(
        X, y, snr_arr, strat, test_size=0.30, random_state=42, stratify=strat
    )
    X_val, X_test, y_val, y_test, snr_val, snr_test, _, _ = train_test_split(
        X_tmp, y_tmp, snr_tmp, strat_tmp, test_size=0.50, random_state=42, stratify=strat_tmp
    )

    out_train = os.path.join(save_dir, f"train_{tag}.npz")
    out_val = os.path.join(save_dir, f"val_{tag}.npz")
    out_test = os.path.join(save_dir, f"test_{tag}.npz")

    np.savez_compressed(out_train, X=X_train, y=y_train, snr_db=snr_train, mod_labels=np.array(classes))
    np.savez_compressed(out_val,   X=X_val,   y=y_val,   snr_db=snr_val,   mod_labels=np.array(classes))
    np.savez_compressed(out_test,  X=X_test,  y=y_test,  snr_db=snr_test,  mod_labels=np.array(classes))

    return {"train": out_train, "val": out_val, "test": out_test}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build modulation dataset")
    parser.add_argument("--examples", type=int, default=200, help="Examples per class per SNR")
    parser.add_argument("--use-multipath", action="store_true", help="Enable multipath channel")
    parser.add_argument("--no-multipath", action="store_true", help="Disable multipath channel")
    parser.add_argument("--save-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--tag", type=str, default="v1", help="Dataset tag suffix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="reports", help="Directory to store logs")
    args = parser.parse_args()

    use_multipath = True
    if args.no_multipath:
        use_multipath = False
    elif args.use_multipath:
        use_multipath = True

    logger = configure_logging(log_dir=args.log_dir, file_prefix="build_dataset")
    logger.info("Starting dataset build")
    logger.info(f"examples_per_class_per_snr={args.examples} | use_multipath={use_multipath} | save_dir={args.save_dir} | tag={args.tag} | seed={args.seed}")

    cfg = SignalConfig()
    paths = build_dataset(
        cfg,
        examples_per_class_per_snr=args.examples,
        use_multipath=use_multipath,
        save_dir=args.save_dir,
        tag=args.tag,
        seed=args.seed,
        logger=logger,
    )
    logger.info(paths)
