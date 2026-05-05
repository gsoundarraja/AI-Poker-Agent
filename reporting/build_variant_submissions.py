import argparse
import json
import os
import shutil
import zipfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")
BUILD_ROOT = os.path.join(ROOT, "submission_variants")

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

VARIANTS = {
    "cfr_blueprint": {
        "belief_search": False,
        "preflop_lookup": False,
    },
    "preflop_lookup": {
        "belief_search": False,
        "preflop_lookup": True,
    },
    "public_belief_search": {
        "belief_search": True,
        "preflop_lookup": False,
    },
    "full_experimental": {
        "belief_search": True,
        "preflop_lookup": True,
    },
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


def fresh_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def copy_tree(build_dir, src_rel, dst_rel):
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(build_dir, dst_rel)
    if not os.path.isdir(src):
        raise RuntimeError("source dir missing: {}".format(src))
    shutil.copytree(src, dst, ignore=_ignore)


def copy_renamed_player(build_dir):
    src = os.path.join(ROOT, "pokeragent.py")
    dst = os.path.join(build_dir, "custom_player.py")
    with open(src) as f:
        text = f.read()
    text = text.replace("PokerAgent", "CustomPlayer")
    with open(dst, "w") as f:
        f.write(text)


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def set_variant_flags(build_dir, config):
    belief_path = os.path.join(build_dir, "data", "belief_search_params.json")
    belief = load_json(belief_path)
    belief["enabled"] = bool(config["belief_search"])
    write_json(belief_path, belief)

    preflop_path = os.path.join(build_dir, "data", "preflop_lookup_params.json")
    preflop = load_json(preflop_path)
    preflop["enabled"] = bool(config["preflop_lookup"])
    write_json(preflop_path, preflop)


def make_zip(build_dir, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, filenames in os.walk(build_dir):
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                inner = os.path.relpath(full, build_dir)
                rel = os.path.join("submission", inner)
                z.write(full, rel)


def build_variant(name, config, tag):
    build_dir = os.path.join(BUILD_ROOT, name)
    fresh_dir(build_dir)
    copy_renamed_player(build_dir)
    copy_tree(build_dir, "agent", "agent")
    copy_tree(build_dir, "data", "data")
    copy_tree(build_dir, "training", "training")
    set_variant_flags(build_dir, config)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    zip_path = os.path.join(CHECKPOINT_DIR, "submission_{}_{}.zip".format(name, tag))
    make_zip(build_dir, zip_path)
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="robust_20260505",
        help="suffix for generated zips",
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma-separated variants: {}".format(",".join(VARIANTS)),
    )
    args = parser.parse_args()

    selected = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in selected if v not in VARIANTS]
    if unknown:
        raise SystemExit("unknown variants: {}".format(",".join(unknown)))

    for name in selected:
        path = build_variant(name, VARIANTS[name], args.tag)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print("{} {:.2f} MB".format(path, size_mb))


if __name__ == "__main__":
    main()
