"""
This function is used to preprocess the dataset. It will downsample the RGB, input depth,
and ground truth depth images to the specified resolution.
"""

import os
from os.path import join
from PIL import Image

import glob
import argparse

from tqdm import tqdm
from multiprocessing import Pool

from creste.datasets.coda_utils import (
    CAMERA_DIR, CAMERA_SUBDIRS, DEPTH_DIR, DEPTH_SUBDIRS)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Downsample frames")
    parser.add_argument("--seqs", nargs="+", type=int, default=[0], help="Sequence indices to process")
    parser.add_argument("--ds", type=int, default=2, help="Downsample factor")
    parser.add_argument("--root_dir", type=str, default="./data/coda_rlang", help="Root directory of the dataset")
    return parser.parse_args()

def process_image_single(inputs):
    infile, outfile, ds_rgb = inputs

    # Read the image
    img = Image.open(infile)
    Hs, Ws = img.size[0] / ds_rgb, img.size[1] / ds_rgb

    interpolation_strategy = Image.NEAREST if "depth" in infile else Image.BILINEAR

    # Apply resizing. We use PIL because it resizes with antialiasing
    img = img.resize((int(Hs), int(Ws)), interpolation_strategy)

    # Save the image
    img.save(outfile)


if __name__ == '__main__':
    print("Preprocessing dataset...")
    args = parse_args()

    root_dir = args.root_dir
    ds = args.ds

    subdirs = []

    seqs = args.seqs

    # Add rectified image subdirectores
    subdirs.extend(
        [join(CAMERA_DIR, subdir, str(seq))
         for seq in seqs for subdir in CAMERA_SUBDIRS]
    )

    subdirs.extend(
        join(f'{DEPTH_DIR}_0_LAIDW_all', str(seq), subdir)
        for subdir in DEPTH_SUBDIRS for seq in seqs
    )

    print(subdirs)

    infile_list = []
    outfile_list = []
    for subdir in subdirs:
        print(f"Processing {subdir}...")

        infiles = glob.glob(join(root_dir, subdir, "*.png"))
        infiles = sorted(infiles, key=lambda x: int(
            x.split('/')[-1].split('.')[0].split('_')[-1]))

        outdir = join(root_dir, f'downsampled_{ds}', subdir)
        outfiles = [join(outdir, os.path.basename(infile))
                    for infile in infiles]

        if not os.path.exists(outdir):
            print(f'Creating directory {outdir}...')
            os.makedirs(outdir)

        infile_list.extend(infiles)
        outfile_list.extend(outfiles)

    print("Processing images...")
    # process_image_single((infile_list[0], outfile_list[0], ds_gt))
    pool = Pool(processes=64)
    for _ in tqdm(pool.imap_unordered(process_image_single, zip(
        infile_list,
        outfile_list,
        [ds] * len(infile_list)
    )), total=len(infile_list)):
        pass
