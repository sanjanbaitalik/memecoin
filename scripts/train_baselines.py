from _bootstrap import bootstrap_path

bootstrap_path()

from prism.pipeline import run_baseline_experiments


if __name__ == "__main__":
    run_baseline_experiments()
