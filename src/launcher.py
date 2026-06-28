import subprocess
import sys


def run_traitement_rep():
    """Exécute traitementRep.py pour générer data/user_profile.json."""
    cmd = [sys.executable, "src/traitementRep.py"]
    print("Lancement de traitementRep.py...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"traitementRep.py a échoué avec le code {proc.returncode}")
    print("traitementRep.py terminé avec succès.")


def run_skill_matcher():
    """Exécute skill_matcher.py pour analyser et recommander jobs."""
    cmd = [sys.executable, "src/skill_matcher.py"]
    print("Lancement de skill_matcher.py...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"skill_matcher.py a échoué avec le code {proc.returncode}")
    print("skill_matcher.py terminé avec succès.")


def run_full_pipeline():
    """Exécute traitementRep.py puis skill_matcher.py."""
    run_traitement_rep()
    run_skill_matcher()


if __name__ == "__main__":
    run_full_pipeline()