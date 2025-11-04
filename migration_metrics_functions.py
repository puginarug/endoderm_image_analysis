# functions to compute migration-related metrics from trajectory data
import numpy as np
import pandas as pd

def compute_turning_angles(df, track_col='track_id', x_col='x_microns', y_col='y_microns', step_col='step', lag=1):
    """
    Compute signed turning angles (radians) between consecutive movement vectors for each track.
    Angles are in range (-pi, pi), where positive means counterclockwise turn.
    """
    if not {track_col, x_col, y_col, step_col}.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {track_col}, {x_col}, {y_col}, {step_col}")

    df_sorted = df.sort_values([track_col, step_col]).copy()
    all_angles = []

    for _, track_df in df_sorted.groupby(track_col):
        x = track_df[x_col].values
        y = track_df[y_col].values

        dx = np.diff(x)
        dy = np.diff(y)

        for i in range(lag, len(dx)):
            v1 = np.array([dx[i - lag], dy[i - lag]])
            v2 = np.array([dx[i], dy[i]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue

            # Normalize
            v1 /= norm1
            v2 /= norm2

            # Compute signed angle using atan2 of cross and dot products
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            dot = np.dot(v1, v2)
            angle = np.arctan2(cross, dot)
            all_angles.append(angle)

    return all_angles


# Primary migration direction and relative angles
def compute_primary_axis(group):
    dx = group['x_microns'].iloc[-1] - group['x_microns'].iloc[0]
    dy = group['y_microns'].iloc[-1] - group['y_microns'].iloc[0]
    return np.arctan2(dy, dx)

def calculate_relative_angles(row):
    angle_diff = row['angle_relative_to_common_axis'] - row['primary_axis']
    return np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize to [-π, π]

def calculate_angle_to_center(row):
    # The angle between the movement vector and the vector pointing to the center
    
    # Vector pointing to the center
    r_x = row['mask_center_x_microns'] - row['prev_x_microns']
    r_y = row['mask_center_y_microns'] - row['prev_y_microns']

    # Movement vector (change in position)
    v_x = row['x_microns'] - row['prev_x_microns'] 
    v_y = row['y_microns'] - row['prev_y_microns'] 

    # v_x = np.cos(row['angle_radians'])
    # v_y = np.sin(row['angle_radians'])

    # Dot product and magnitudes
    dot_product = r_x * v_x + r_y * v_y
    magnitude_r = np.sqrt(r_x**2 + r_y**2)
    magnitude_v = np.sqrt(v_x**2 + v_y**2)
    
    # Compute the angle (in radians)
    cos_theta = dot_product / (magnitude_r * magnitude_v)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip for numerical stability
    
    # Determine the sign using the cross product
    cross_product = r_x * v_y - r_y * v_x
    if cross_product < 0:
        theta = -theta  # Clockwise angles are negative
    
    return theta # angle in radians

# MSD
def calculate_msd(df, x_col='x_microns', y_col='y_microns', max_lag=None):
    """
    Calculate the Mean Squared Displacement (MSD) for particle tracks in a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing particle tracking data. Must include columns:
        'track_id', 'x_microns', and 'y_microns'.
    dt : int
        The time interval (in frames or time units) between positions for MSD calculation.
        Used for scaling the 'dt' column in the output.
    max_lag : int, optional
        The maximum lag (number of steps) to compute MSD for. If None, uses the maximum
        possible lag based on the longest track.
    Returns
    -------
    msd_df : pandas.DataFrame
        DataFrame with columns:
            - 'msd': Mean squared displacement for each lag.
            - 'msd_std': Standard deviation of squared displacements for each lag.
            - 'lag': Lag values (number of steps).
            - 'dt': Lag values converted to time units (lag * input lag parameter).
    Notes
    -----
    - The function groups data by 'track_id' and computes MSD for each track, then averages across all tracks.
    - If no values are available for a given lag, NaN is returned for that lag.
    """

    # Ensure required columns are present
    required_cols = {'track_id', 'x_microns', 'y_microns'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
    
    # Group by track_id
    tracks = df.groupby('track_id')
    n_steps = df.groupby('track_id').size().max()
    
    if max_lag is None:
        max_lag = n_steps - 1
    else:
        max_lag = min(max_lag, n_steps - 1)
    
    msd_values = [[] for _ in range(max_lag)]
    
    for track_id, track in tracks:
        x = track[x_col].values
        y = track[y_col].values
        n = len(x)
        
        for lag in range(1, min(n, max_lag + 1)):
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            squared_disp = dx**2 + dy**2
            msd_values[lag-1].extend(squared_disp)
    
    msd_mean = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in msd_values])
    msd_std = np.array([np.std(vals) if len(vals) > 0 else np.nan for vals in msd_values])
    msd_sem = np.array([np.std(vals)/np.sqrt(len(vals)) if len(vals) > 0 else np.nan 
                        for vals in msd_values])
    n_samples = np.array([len(vals) for vals in msd_values])

    # Calculate time resolution from the actual data
    time_steps = df.groupby('step')['t'].first().sort_index()
    time_res = time_steps.iloc[1] - time_steps.iloc[0]

    msd_df = pd.DataFrame({
        'msd': msd_mean,
        'msd_std': msd_std,
        'msd_sem': msd_sem,
        'n': n_samples,
        'lag': np.arange(1, max_lag + 1),
        'dt': np.arange(1, max_lag + 1) * time_res  # Convert to minutes
    })
    
    return msd_df


def calculate_autocorrelation(df, max_lag=None, directional=True):
    """
    Robust DACF/VACF computation with safe handling when no valid samples exist.
    Returns DataFrame with columns: {dacf|vacf}, {dacf|vacf}_std, {dacf|vacf}_sem, n, lag, dt
    """
    required_cols = {'track_id', 'step', 'v_x', 'v_y', 't'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # pivot (tracks x steps). Missing steps become NaN.
    df_vx = df.pivot(index='track_id', columns='step', values='v_x')
    df_vy = df.pivot(index='track_id', columns='step', values='v_y')
    Vx = df_vx.values
    Vy = df_vy.values

    n_particles, n_steps = Vx.shape
    max_possible_lag = n_steps if max_lag is None else min(n_steps, max_lag)

    acorr_vals = np.full(max_possible_lag, np.nan, dtype=float)
    acorr_stds = np.full(max_possible_lag, np.nan, dtype=float)
    acorr_sems = np.full(max_possible_lag, np.nan, dtype=float)
    n_samples = np.zeros(max_possible_lag, dtype=int)

    for dt in range(max_possible_lag):
        # Align arrays for lag dt
        v1x = Vx[:, :n_steps-dt]
        v2x = Vx[:, dt:]
        v1y = Vy[:, :n_steps-dt]
        v2y = Vy[:, dt:]

        # elementwise dot product (NaNs propagate)
        dot = v1x * v2x + v1y * v2y

        if directional:
            mag1 = np.sqrt(v1x**2 + v1y**2)
            mag2 = np.sqrt(v2x**2 + v2y**2)
            # valid where both magnitudes > 0 and neither component is NaN
            valid_mask = (mag1 > 0) & (mag2 > 0) & (~np.isnan(mag1)) & (~np.isnan(mag2))
            n_valid = int(np.count_nonzero(valid_mask))
            n_samples[dt] = n_valid

            if n_valid == 0:
                # No valid pairs for this lag
                acorr_vals[dt] = np.nan
                acorr_stds[dt] = np.nan
                acorr_sems[dt] = np.nan
            else:
                # compute normalized dot only for valid entries
                norm_dot = np.empty_like(dot)
                norm_dot[:] = np.nan
                norm_dot[valid_mask] = (dot[valid_mask] /
                                        (mag1[valid_mask] * mag2[valid_mask]))
                # safe nan-aggregations (should have at least n_valid non-nan)
                acorr_vals[dt] = np.nanmean(norm_dot)
                acorr_stds[dt] = np.nanstd(norm_dot)
                acorr_sems[dt] = acorr_stds[dt] / np.sqrt(n_valid)
        else:
            # VACF (no normalization). Count non-NaN dot entries
            valid_mask = ~np.isnan(dot)
            n_valid = int(np.count_nonzero(valid_mask))
            n_samples[dt] = n_valid

            if n_valid == 0:
                acorr_vals[dt] = np.nan
                acorr_stds[dt] = np.nan
                acorr_sems[dt] = np.nan
            else:
                values = dot[valid_mask]
                acorr_vals[dt] = np.mean(values)
                acorr_stds[dt] = np.std(values)
                acorr_sems[dt] = acorr_stds[dt] / np.sqrt(n_valid)

    lags = np.arange(max_possible_lag)

    column_name = 'dacf' if directional else 'vacf'
    acorr_df = pd.DataFrame({
        column_name: acorr_vals,
        f'{column_name}_std': acorr_stds,
        f'{column_name}_sem': acorr_sems,
        'n': n_samples,
        'lag': lags
    })

    # compute time resolution safely
    time_steps = df.groupby('step')['t'].first().sort_index()
    if len(time_steps) >= 2:
        time_res = time_steps.iloc[1] - time_steps.iloc[0]
    else:
        time_res = np.nan  # unknown or single time point

    acorr_df['dt'] = acorr_df['lag'] * time_res

    return acorr_df


# --- Robust weighted aggregation & pipeline ---
def compute_msd_dacf_per_movie(df, x_col='x_microns', y_col='y_microns', max_lag=None):
    """
    Compute MSD and DACF per movie and produce weighted aggregates.
    Safely handles movies with no valid results and lags with zero total weight.
    """

    msd_results = []
    dacf_results = []

    for movie_id, df_movie in df.groupby("file"):
        try:
            msd_df = calculate_msd(df_movie, x_col=x_col, y_col=y_col, max_lag=max_lag)
            msd_df = msd_df.copy()
            msd_df["file"] = movie_id
            msd_results.append(msd_df)
        except Exception as e:
            print(f"⚠️ Skipped MSD for movie {movie_id} due to error: {e}")

        try:
            dacf_df = calculate_autocorrelation(df_movie, max_lag=max_lag, directional=True)
            dacf_df = dacf_df.copy()
            dacf_df["file"] = movie_id
            dacf_results.append(dacf_df)
        except Exception as e:
            print(f"⚠️ Skipped DACF for movie {movie_id} due to error: {e}")

    # If no results, return empty DataFrames with expected columns
    if len(msd_results) == 0:
        msd_summary = pd.DataFrame(columns=['lag', 'msd_mean', 'msd_std', 'msd_sem', 'dt', 'n_total'])
    else:
        msd_all = pd.concat(msd_results, ignore_index=True)
        msd_all_renamed = msd_all.rename(columns={'msd': 'value'})

        # weighted aggregation function (robust)
        def weighted_stats(group):
            values = group['value'].to_numpy(dtype=float)
            dt = group['dt'].iloc[0] if 'dt' in group.columns else np.nan

            if 'n' in group.columns:
                weights = group['n'].to_numpy(dtype=float)
                # consider only entries with positive weight and non-nan values
                mask = (weights > 0) & (~np.isnan(values))
                if mask.sum() == 0:
                    # fallback to unweighted using non-nan values
                    valid = ~np.isnan(values)
                    if valid.sum() == 0:
                        return pd.Series({'mean': np.nan, 'std': np.nan, 'sem': np.nan, 'dt': dt, 'n_total': 0})
                    vals = values[valid]
                    un_n = valid.sum()
                    return pd.Series({
                        'mean': vals.mean(),
                        'std': vals.std(ddof=0),
                        'sem': vals.std(ddof=0) / np.sqrt(un_n),
                        'dt': dt,
                        'n_total': int(un_n)
                    })
                w = weights[mask]
                v = values[mask]
                wsum = w.sum()
                wmean = np.average(v, weights=w)
                wvar = np.average((v - wmean)**2, weights=w)
                wstd = np.sqrt(wvar)
                denom = (w**2).sum()
                n_eff = (wsum**2 / denom) if denom > 0 else np.nan
                wsem = (wstd / np.sqrt(n_eff)) if (n_eff is not None and n_eff > 0) else np.nan
                return pd.Series({'mean': wmean, 'std': wstd, 'sem': wsem, 'dt': dt, 'n_total': float(wsum)})
            else:
                # no weights -> unweighted
                valid = ~np.isnan(values)
                if valid.sum() == 0:
                    return pd.Series({'mean': np.nan, 'std': np.nan, 'sem': np.nan, 'dt': dt, 'n_total': 0})
                vals = values[valid]
                n = valid.sum()
                return pd.Series({'mean': vals.mean(), 'std': vals.std(ddof=0), 'sem': vals.std(ddof=0) / np.sqrt(n), 'dt': dt, 'n_total': int(n)})

        # Use include_groups=False to avoid the FutureWarning (Pandas >= 2.4)
        try:
            msd_summary = msd_all_renamed.groupby("lag").apply(weighted_stats, include_groups=False).reset_index()
        except TypeError:
            # older pandas versions may not accept include_groups, fallback:
            msd_summary = msd_all_renamed.groupby("lag").apply(weighted_stats).reset_index()

        msd_summary.columns = ['lag', 'msd_mean', 'msd_std', 'msd_sem', 'dt', 'n_total']

    # DACF side
    if len(dacf_results) == 0:
        dacf_summary = pd.DataFrame(columns=['lag', 'dacf_mean', 'dacf_std', 'dacf_sem', 'dt', 'n_total'])
    else:
        dacf_all = pd.concat(dacf_results, ignore_index=True)
        dacf_all_renamed = dacf_all.rename(columns={'dacf': 'value'})

        def weighted_stats_dacf(group):
            # reuse same logic as weighted_stats but keep a separate name for clarity
            return weighted_stats(group)

        try:
            dacf_summary = dacf_all_renamed.groupby("lag").apply(weighted_stats_dacf, include_groups=False).reset_index()
        except TypeError:
            dacf_summary = dacf_all_renamed.groupby("lag").apply(weighted_stats_dacf).reset_index()

        dacf_summary.columns = ['lag', 'dacf_mean', 'dacf_std', 'dacf_sem', 'dt', 'n_total']

    return msd_summary, dacf_summary
# step length distribution
def compute_step_length_distribution(df, step_col='step', x_col='x_microns', y_col='y_microns', turning_angle_threshold=np.pi/4):
    """
    Compute the distribution of step lengths for particle trajectories.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing particle tracking data. Must include columns:
        'track_id', 'step', 'x_microns', and 'y_microns'.
    step_col : str
        Name of the column representing time steps.
    x_col : str
        Name of the column representing x positions.
    y_col : str
        Name of the column representing y positions.
    turning_angle_threshold : float
        Threshold (in radians) to classify steps as 'straight' or 'turning'.
    Returns
    -------
    step_lengths_df : pandas.DataFrame
        DataFrame with columns:
            - 'step_length': Length of each step.
            - 'turning_angle': Turning angle (radians) at each step.
            - 'step_type': 'straight' or 'turning' based on the turning angle threshold.
    """
    # Ensure required columns are present
    required_cols = {'track_id', step_col, x_col, y_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
    
    df_sorted = df.sort_values(['track_id', step_col]).copy()
    step_lengths = []
    turning_angles = []
    step_types = []

    for _, track_df in df_sorted.groupby('track_id'):
        x = track_df[x_col].values
        y = track_df[y_col].values

        dx = np.diff(x)
        dy = np.diff(y)

        for i in range(1, len(dx)):
            # Step length
            length = np.sqrt(dx[i]**2 + dy[i]**2)
            step_lengths.append(length)

            # Compute turning angle
            v1 = np.array([dx[i - 1], dy[i - 1]])
            v2 = np.array([dx[i], dy[i]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                angle = 0.0
            else:
                v1 /= norm1
                v2 /= norm2
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                dot = np.dot(v1, v2)
                angle = np.arctan2(cross, dot)
            turning_angles.append(angle)

            # Classify step type
            if abs(angle) < turning_angle_threshold:
                step_types.append('straight')
            else:
                step_types.append('turning')

    step_lengths_df = pd.DataFrame({
        'step_length': step_lengths,
        'turning_angle': turning_angles,
        'step_type': step_types
    })

    return step_lengths_df