###############################
###############################
##                                                          ##
##          src/feature_anal_eng.py          ##
##                                                          ##
###############################
###############################



from __future__ import annotations
import numpy as np
import pandas as pd



def add_innerness_and_outerness(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """
    Return a copy of `df` with two new columns quantifying the 'innerness' and
    'outerness' of flagged sequences in `flag_col`.

    The function treats entries outside the DataFrame as the *opposite* flag value,
    ensuring that contiguous blocks at the boundaries are properly closed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a binary (0/1) flag column.
    flag_col : str
        Name of the column with 0/1 entries.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with two new tuple-valued columns:
        - `<flag_col>_innerness`: (k, l) distances inside 1-blocks
        - `<flag_col>_outerness`: (k, l) distances inside 0-blocks
          (both follow the same virtual-boundary rule)
    """
    if flag_col not in df.columns:
        raise KeyError(f"Column '{flag_col}' not found in DataFrame.")

    flags = df[flag_col].to_numpy()
    n = len(flags)
    idx = np.arange(n)

    # --- Helper: compute (k, l) for a given target value ---
    def compute_innerness_for_value(target_value: int) -> np.ndarray:
        # Mark positions of the opposite value
        opposite_mask = flags != target_value
        sparse_idx = np.where(opposite_mask, idx, np.nan)

        # Forward and backward fill to find nearest opposites
        prev_opposite = pd.Series(sparse_idx).ffill().to_numpy()
        next_opposite = pd.Series(sparse_idx).bfill().to_numpy()

        # Apply virtual opposite values outside frame
        prev_opposite = np.where(np.isnan(prev_opposite), -1, prev_opposite)
        next_opposite = np.where(np.isnan(next_opposite), n, next_opposite)

        # Compute distances to last and next opposite values
        k = idx - prev_opposite
        l = next_opposite - idx

        # Zero distances outside target blocks
        pairs = np.stack([k, l], axis=1)
        pairs[flags != target_value] = (0, 0)
        return pairs.astype(int)

    # Compute both
    inner_pairs = compute_innerness_for_value(1)
    outer_pairs = compute_innerness_for_value(0)

    # Return a copy with two new tuple-valued columns
    df_copy = df.copy()
    df_copy[f"{flag_col}_innerness"] = list(map(tuple, inner_pairs))
    df_copy[f"{flag_col}_outerness"] = list(map(tuple, outer_pairs))

    return df_copy
