from _bootstrap import bootstrap_path

bootstrap_path()

from prism.pipeline import run_significance_tests


if __name__ == "__main__":
    run_significance_tests()
