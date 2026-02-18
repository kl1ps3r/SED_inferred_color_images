import argparse
import os
import numpy as np
import pandas as pd
import time

# -----------------------------
# Editable distributions/config
# -----------------------------
POISSON_LAMBDA = 0.6
RNG_SEED = int(time.time())  # Use current time as seed for variability; can be set to a fixed value for reproducibility

DISTRIBUTIONS = {
    "effective_radius_kpc": {"dist": "normal", "mean": 3.5, "std": 1.5, "min": 0.1},
    "sersic_index": {"dist": "normal", "mean": 3.0, "std": 1.0, "min": 0.3},
    "axis_ratio": {"dist": "axis_ratio_e1e2", "sigma": 0.2, "min": 0.1, "max": 1.0},
    "AB_magnitude": {"dist": "normal", "mean": 21.5, "std": 0.7},
    "angle": {"dist": "uniform", "low": 0.0, "high": 2 * np.pi},
    "redshift": {"dist": "uniform", "low": 0.1, "high": 0.7},
}

POSITION_DISTRIBUTION = {
    "min_radius": 2.5,
    "max_radius": 11.5,
}

OUTPUT_DEFAULT = "./testing_additional_gals/additional_gals_params.csv"


def _sample_axis_ratio(rng, sigma, min_val=0.1, max_val=1.0, size=1):
    e1 = rng.normal(0, sigma, size)
    e2 = rng.normal(0, sigma, size)

    c = np.sqrt(e1**2 + e2**2)
    q = (1 - c) / (1 + c)
    q = np.clip(q, min_val, max_val)

    assert not np.isnan(q).any(), "Axis ratio sampling resulted in NaN values."
    return q


def _sample_edge_annulus(rng, min_radius, max_radius, size=1):
    u = rng.uniform(0.0, 1.0, size)
    r = np.sqrt(u * (max_radius**2 - min_radius**2) + min_radius**2)
    theta = rng.uniform(0.0, 2 * np.pi, size)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _sample_param(rng, spec, size=1):
    dist = spec["dist"]
    if dist == "normal":
        vals = rng.normal(spec["mean"], spec["std"], size)
        if "min" in spec:
            vals = np.clip(vals, spec["min"], None)
        if "max" in spec:
            vals = np.clip(vals, None, spec["max"])
        return vals
    if dist == "uniform":
        return rng.uniform(spec["low"], spec["high"], size)
    if dist == "axis_ratio_e1e2":
        return _sample_axis_ratio(rng, spec["sigma"], spec.get("min", 0.1), spec.get("max", 1.0), size)
    raise ValueError(f"Unsupported distribution: {dist}")


def generate_additional_galaxies(num_images, rng, poisson_lambda=None):
    rows = []
    lam = POISSON_LAMBDA if poisson_lambda is None else poisson_lambda
    for image_num in range(num_images):
        n_galaxies = rng.poisson(lam)
        if n_galaxies == 0:
            continue
        if n_galaxies > 5:
            n_galaxies = 5  # Cap at 5 galaxies per image to avoid overcrowding

        effective_radius_kpc = _sample_param(rng, DISTRIBUTIONS["effective_radius_kpc"], n_galaxies)
        sersic_index = _sample_param(rng, DISTRIBUTIONS["sersic_index"], n_galaxies)
        axis_ratio = _sample_param(rng, DISTRIBUTIONS["axis_ratio"], n_galaxies)
        ab_mag = _sample_param(rng, DISTRIBUTIONS["AB_magnitude"], n_galaxies)
        pos_x, pos_y = _sample_edge_annulus(
            rng,
            POSITION_DISTRIBUTION["min_radius"],
            POSITION_DISTRIBUTION["max_radius"],
            n_galaxies,
        )
        angle = _sample_param(rng, DISTRIBUTIONS["angle"], n_galaxies)
        redshift = _sample_param(rng, DISTRIBUTIONS["redshift"], n_galaxies)

        for j in range(n_galaxies):
            rows.append(
                {
                    "effective_radius_kpc": effective_radius_kpc[j],
                    "sersic_index": sersic_index[j],
                    "axis_ratio": axis_ratio[j],
                    "AB_magnitude": ab_mag[j],
                    "pos_x": pos_x[j],
                    "pos_y": pos_y[j],
                    "angle": angle[j],
                    "redshift": redshift[j],
                    "image_num": image_num,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate additional edge-galaxy parameter CSV.")
    parser.add_argument("--num_images", type=int, required=True, help="Number of images to generate rows for.")
    parser.add_argument("--output_path", type=str, default=OUTPUT_DEFAULT, help="Output CSV path.")
    parser.add_argument("--poisson_lambda", type=float, default=POISSON_LAMBDA, help="Poisson lambda for galaxy count per image.")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = generate_additional_galaxies(args.num_images, rng, poisson_lambda=args.poisson_lambda)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Wrote {len(df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
