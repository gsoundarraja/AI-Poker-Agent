import os
import shutil
import sys
import zipfile


ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(ROOT, "submission")
ZIP_PATH = os.path.join(ROOT, "submission.zip")


def fresh_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

EXCLUDE_FILENAMES = {
    ".DS_Store",
    "training_curve.csv",
    "cfr_training_curve.csv",
    "cfr_training.lock",
    "eval_weights.json",
    "preflop_hs.pkl",
}

EXCLUDE_DIRNAMES = {
    "checkpoints",
}


def _ignore(_dir, names):
    return [
        n for n in names
        if (
            n in EXCLUDE_FILENAMES
            or n in EXCLUDE_DIRNAMES
            or n.startswith("__pycache__")
            or n.endswith(".bak")
            or n.endswith(".zip")
        )
    ]


def copy_tree(src_rel, dst_rel):
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(BUILD_DIR, dst_rel)
    if not os.path.isdir(src):
        sys.exit("ERROR: source dir missing: {}".format(src))
    shutil.copytree(src, dst, ignore=_ignore)


def copy_renamed_player():
    src = os.path.join(ROOT, "pokeragent.py")
    dst = os.path.join(BUILD_DIR, "custom_player.py")
    with open(src) as f:
        text = f.read()
    text = text.replace("PokerAgent", "CustomPlayer")
    with open(dst, "w") as f:
        f.write(text)


def make_zip():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, filenames in os.walk(BUILD_DIR):
            for fname in filenames:
                full = os.path.join(dirpath, fname)

                rel = os.path.relpath(full, ROOT)
                z.write(full, rel)


def main():
    print("[1/4] Resetting", BUILD_DIR)
    fresh_dir(BUILD_DIR)

    print("[2/4] Copying pokeragent.py -> submission/custom_player.py "
          "(class PokerAgent -> CustomPlayer)")
    copy_renamed_player()

    print("[3/4] Copying helper packages")
    copy_tree("agent", "agent")
    copy_tree("data", "data")
    copy_tree("training", "training")

    print("[4/4] Building zip:", ZIP_PATH)
    make_zip()

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print("Done. {} ({:.2f} MB)".format(ZIP_PATH, size_mb))
    print("Upload this file at http://cs683.cs.umass.edu")


if __name__ == "__main__":
    main()
