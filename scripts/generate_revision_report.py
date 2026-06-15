from _bootstrap import bootstrap_path

bootstrap_path()

from prism.pipeline import generate_experiment_registry, generate_revision_reports


if __name__ == "__main__":
    generate_experiment_registry()
    generate_revision_reports()
