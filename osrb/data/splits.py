from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def select_open_set_classes(
    train_keys: Sequence[str],
    eval_keys: Sequence[str],
    split_mode: str = "fixed",
    num_known: Optional[int] = None,
    num_unknown: Optional[int] = None,
    known_classes: Optional[Sequence[str]] = None,
    unknown_classes: Optional[Sequence[str]] = None,
    split_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Determine known and unknown class lists under 'fixed', 'random', or 'explicit' protocols.

    - 'fixed': Uses train_keys as known (or first num_known) and remaining eval_keys as unknown.
    - 'random': Randomly samples num_known known classes from train_keys and num_unknown unknown
                classes from remaining eval_keys using split_seed.
    - 'explicit' / 'curated': Uses user-provided known_classes and unknown_classes lists.
    """
    mode = split_mode.lower()
    train_list = list(train_keys)
    eval_list = list(eval_keys)

    if mode in ("explicit", "curated"):
        if not known_classes or not unknown_classes:
            raise ValueError("split_mode='explicit' requires both known_classes and unknown_classes.")
        k_list = list(known_classes)
        u_list = list(unknown_classes)
        if set(k_list) & set(u_list):
            raise ValueError(f"Overlap between known_classes and unknown_classes: {set(k_list) & set(u_list)}")
        return k_list, u_list

    if mode == "fixed":
        k_list = list(known_classes) if known_classes else list(train_list)
        if num_known is not None and num_known < len(k_list):
            k_list = k_list[:num_known]
        u_candidates = [k for k in eval_list if k not in k_list]
        u_list = list(unknown_classes) if unknown_classes else u_candidates
        if num_unknown is not None and num_unknown < len(u_list):
            u_list = u_list[:num_unknown]
        return k_list, u_list

    if mode == "random":
        rng = np.random.RandomState(split_seed)
        n_k = num_known if num_known is not None else len(train_list)
        if n_k > len(train_list):
            raise ValueError(f"Requested num_known={n_k} exceeds available training classes ({len(train_list)}).")
        perm_train = rng.permutation(len(train_list))
        k_list = [train_list[i] for i in perm_train[:n_k]]

        u_candidates = [k for k in eval_list if k not in k_list]
        n_u = num_unknown if num_unknown is not None else len(u_candidates)
        if n_u > len(u_candidates):
            raise ValueError(
                f"Requested num_unknown={n_u} exceeds remaining unknown candidates ({len(u_candidates)})."
            )
        perm_u = rng.permutation(len(u_candidates))
        u_list = [u_candidates[i] for i in perm_u[:n_u]]
        return k_list, u_list

    raise ValueError(f"Unsupported split_mode '{split_mode}'. Choose from ['fixed', 'random', 'explicit'].")
