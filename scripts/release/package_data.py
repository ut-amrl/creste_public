import os
import random
import shutil
import zipfile
from pathlib import Path
import argparse

random.seed(42)  # For reproducibility

def copy_relative(src_root, rel_path, dst_root):
    src = os.path.join(src_root, rel_path)
    dst = os.path.join(dst_root, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def get_seq_frame_pairs_from_counterfactuals(counterfactuals_dir):
    pairs = []
    for seq_dir in Path(counterfactuals_dir).iterdir():
        if not seq_dir.is_dir():
            continue
        for file in seq_dir.glob("*.pkl"):
            frame_id = file.stem  # filename without .pkl
            if frame_id.isdigit():
                pairs.append((seq_dir.name, int(frame_id)))
    return pairs

def main(dataset_root, txt_file, N, W, output_zip):
    dataset_root = Path(dataset_root)
    temp_dir = Path("temp_export")
    temp_dir.mkdir(exist_ok=True)
    W_half = W // 2

    # Get selected sequence/frame_id pairs
    if txt_file:
        with open(txt_file, "r") as f:
            lines = [line.strip().split() for line in f if line.strip()]
            selected = random.sample(lines, N)
    else:
        counterfactuals_dir = dataset_root / "counterfactuals"
        all_pairs = get_seq_frame_pairs_from_counterfactuals(counterfactuals_dir)
        if len(all_pairs) < N:
            raise ValueError(f"Not enough counterfactual pairs to sample {N}. Found {len(all_pairs)}.")
        selected = random.sample(all_pairs, N)

    # Write selected seq/frame pairs to full.txt

    # Sort selected pairs by seq_id and frame_id
    selected = sorted(selected, key=lambda x: (int(x[0]), int(x[1])))

    seq_frame_pairs = []
    for seq_id, frame_id in selected:
        seq_id = str(seq_id)
        frame_id = int(frame_id)
        frame_range = range(frame_id - W_half, frame_id + W_half + 1)

        # Copy 2d_rect frames
        for cam in ["cam0", "cam1"]:
            for fid in frame_range:
                jpg_name = f"2d_rect_{cam}_{seq_id}_{fid}.jpg"
                rel_path = f"2d_rect/{cam}/{seq_id}/{jpg_name}"

                # info_name = f"{fid}.pkl"
                # info_rel_path = f"infos/{cam}/{seq_id}/{info_name}"
                try:
                    copy_relative(dataset_root, rel_path, temp_dir)
                    # copy_relative(dataset_root, info_rel_path, temp_dir)
                except FileNotFoundError:
                    print(f"Missing: {rel_path}")

        # Copy 3d_raw frames (os1 only)
        for i, fid in enumerate(frame_range):
            bin_name = f"3d_raw_os1_{seq_id}_{fid}.bin"
            rel_path = f"3d_raw/os1/{seq_id}/{bin_name}"
            try:
                copy_relative(dataset_root, rel_path, temp_dir)
            except FileNotFoundError:
                print(f"Missing: {rel_path}")
            if i % 5 == 0: # Add every 5th frame
                seq_frame_pairs.append(f"{seq_id} {fid}")

        # Copy matching counterfactual
        pkl_path = f"counterfactuals/{seq_id}/{frame_id}.pkl"
        try:
            copy_relative(dataset_root, pkl_path, temp_dir)
        except FileNotFoundError:
            print(f"Missing: {pkl_path}")


    split_dir = temp_dir / "splits" / "mini"
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "full.txt", "w") as f:
        for seq_frame_pair in seq_frame_pairs:
            f.write(f"{seq_frame_pair}\n")

    # Always copy full folders
    for folder in ["calibrations", "timestamps", "poses/dense"]:
        full_src = dataset_root / folder
        full_dst = temp_dir / folder
        if full_src.exists():
            shutil.copytree(full_src, full_dst, dirs_exist_ok=True)

    # Create zip
    creste_root = temp_dir / "creste"
    creste_root.mkdir(parents=True, exist_ok=True)

    # Move all contents of temp_dir into creste/
    for item in list(temp_dir.iterdir()):
        if item.name == "creste":
            continue
        shutil.move(str(item), creste_root / item.name)

    # Create zip with "creste/" as top-level
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', root_dir=temp_dir, base_dir="creste")

    # Cleanup temp folder
    shutil.rmtree(temp_dir)
    print(f"Created zip: {output_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to creste_rlang root dir")
    parser.add_argument("--txt_file", required=False, help="Optional path to .txt file with seq_id and frame_id")
    parser.add_argument("--N", type=int, required=True, help="Number of entries to sample")
    parser.add_argument("--W", type=int, required=True, help="Window size (W/2 before and after)")
    parser.add_argument("--output_zip", required=True, help="Output zip filename")

    args = parser.parse_args()
    main(args.dataset_root, args.txt_file, args.N, args.W, args.output_zip)
