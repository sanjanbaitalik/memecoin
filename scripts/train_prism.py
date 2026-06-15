from _bootstrap import bootstrap_path

bootstrap_path()

from prism.pipeline import run_prism_and_ablation


if __name__ == "__main__":
    run_prism_and_ablation()
