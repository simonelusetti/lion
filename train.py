from lion.experiments import main

if __name__ == "__main__":
    import sys
    from forge import run

    raise SystemExit(run(None, sys.argv[1:]))
