from __future__ import annotations

import argparse
from pathlib import Path

from labeler.csv_logger import CsvDecisionLogger
from labeler.loaders import MultiHduFitsLoader, SeparateFilesByStemLoader
from labeler.ui import FitsLabelerApp


def parse_hdu_map(raw: str) -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for item in raw.split(","):
        k, v = item.split(":", 1)
        v = v.strip()
        out[k.strip()] = int(v) if v.isdigit() else v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tkinter FITS binary labeler")
    parser.add_argument("--images-root", type=Path, required=True, help="Folder containing FITS data")
    parser.add_argument("--csv-path", type=Path, default=Path("labels.csv"), help="Output labels CSV")
    parser.add_argument("--loader", choices=["separate", "multi-hdu"], default="multi-hdu")
    parser.add_argument(
        "--skip-handled",
        action="store_true",
        help="Skip samples already present in CSV (if CSV exists and has matching sample IDs)",
    )

    parser.add_argument("--true-label-text", default="True")
    parser.add_argument("--false-label-text", default="False")

    parser.add_argument("--vis-suffix", default="_VIS.fits")
    parser.add_argument("--y-suffix", default="_NIR_Y.fits")
    parser.add_argument("--j-suffix", default="_NIR_J.fits")
    parser.add_argument("--h-suffix", default="_NIR_H.fits")

    parser.add_argument("--single-pattern", default="*.fits", help="Glob pattern for multi-hdu mode")
    parser.add_argument(
        "--hdu-map",
        default="VIS:1,Y:2,J:3,H:4",
        help="Comma-separated band:HDU mapping for multi-hdu mode",
    )

    args = parser.parse_args()

    if args.loader == "separate":
        loader = SeparateFilesByStemLoader(
            root_dir=args.images_root,
            band_suffixes={
                "VIS": args.vis_suffix,
                "Y": args.y_suffix,
                "J": args.j_suffix,
                "H": args.h_suffix,
            },
        )
    else:
        loader = MultiHduFitsLoader(
            root_dir=args.images_root,
            pattern=args.single_pattern,
            hdu_map=parse_hdu_map(args.hdu_map),
        )

    logger = CsvDecisionLogger(args.csv_path)
    app = FitsLabelerApp(
        loader=loader,
        logger=logger,
        true_label_text=args.true_label_text,
        false_label_text=args.false_label_text,
        skip_previously_labeled=args.skip_handled,
    )
    app.run()


if __name__ == "__main__":
    main()
