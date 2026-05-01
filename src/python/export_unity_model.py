from __future__ import annotations

import argparse

from unity_export import export_run_to_unity_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a training run folder into a Unity-friendly NCA JSON export.")
    parser.add_argument("run_dirs", nargs="+", help="One or more output run directories containing weights.json/config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dirs:
        output_path = export_run_to_unity_json(run_dir)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
