from _bootstrap import bootstrap_path

bootstrap_path()

from prism.pipeline import run_audit


if __name__ == "__main__":
    run_audit()
