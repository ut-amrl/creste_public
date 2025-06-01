"""
This functions generates ground truth SAM masks in the image and BEV space. This module
will aggegates static masks across frame to provide a dense ground truth mask for each
frame.

SAM masks for images are generates for 2x downsampled images.
Ground truth static/dynamic image masks are generated for 2x downsampled images
"""

import os
from os.path import join
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import copy
import json

import numpy as np
from PIL import Image
import cv2

# CUDA RELATED IMPORTS
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

from scripts.preprocessing.build_dense_depth import get_frames_from_json
from creste.datasets.coda_utils import CAMERA_DIR, frame2fn, fn2frame, SAM_DYNAMIC_LABEL_MAP
import creste.utils.visualization as vis

# SAM2 related imports
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from scripts.preprocessing.sam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from scripts.preprocessing.sam2_utils.track_utils import sample_points_from_masks
from scripts.preprocessing.sam2_utils.video_utils import create_video_from_images
from scripts.preprocessing.sam2_utils.common_utils import CommonUtils
from transformers import SamModel, SamProcessor


os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

# print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
# Reduce number of torch threads
torch.set_num_threads(6)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build semantic map from point clouds')
    parser.add_argument(
        '--indir', type=str, default='./data/creste/downsampled_2', help="Path to root directory")
    parser.add_argument(
        '--outdir', type=str, default='./data/creste/downsampled_2/sam2', help="Path to output directory")
    parser.add_argument('--skip_factor', type=int, default=5,
                        help='Number of frames to skip for aggregation')
    parser.add_argument('--camids', nargs='+', type=str, default=['cam0'],
                        help='List of cameras to process (e.g. cam0 cam1)')
    parser.add_argument('--skip_sequences', nargs='+', type=int, default=[],
                        help='List of sequences to skip')
    parser.add_argument('--img_ds', type=int, default=2,
                        help='Downsample factor for images')
    parser.add_argument('--num_chunks', type=int, default=8,
                        help='Number of chunks to split dataset into')
    parser.add_argument('--chunk_idx', type=int, default=0,
                        help='Index of chunk to process')
    parser.add_argument('--mask_type', type=str, default='static',
                        help='Type of mask parser to use [static, dynamic]')
    args = parser.parse_args()

    return args


def filter_segment_mask(mask_batch):
    """
    Inputs:
        mask - list of torch tensors [Z, H, W] of boolean masks. Masks are ordered by confidence.
    """
    B, Z, H, W = mask_batch.shape
    mask_batch_filtered = torch.zeros(
        (B, H, W), dtype=torch.long, device=mask_batch[0].device)
    for mask_idx, mask in enumerate(mask_batch):
        Z = mask.shape[0]
        bool_masks_ids = torch.arange(
            0, Z, dtype=torch.long, device=mask.device).view(Z, 1, 1)
        img_masks = mask.long() * bool_masks_ids
        mask_batch_filtered[mask_idx] = torch.argmax(
            img_masks, dim=0).unsqueeze(0)

    return mask_batch_filtered


def make_labels_contiguous_vectorized(labels):
    """
    Convert a tensor of shape 1xHxW containing label indices to a tensor of the same shape
    with contiguous labels starting from 0, using a vectorized approach.

    Args:
    labels (torch.Tensor): Input tensor of shape 1xHxW where each value is a label index.

    Returns:
    torch.Tensor: Tensor of the same shape with contiguous label indices.
    """
    # Flatten the tensor to get all labels as a 1D array
    unique_labels, new_label_indices = torch.unique(
        labels, sorted=True, return_inverse=True)

    # Reshape the new_label_indices back to the original shape of the labels tensor
    new_labels = new_label_indices.reshape(labels.shape)

    return new_labels


class SequentialDataset(Dataset):
    def __init__(self,
                 outdir_dict,
                 modality,
                 camid,
                 frame_infos):
        self.indir = outdir_dict["raw"]
        self.outdir_dict = outdir_dict
        self.modality = modality
        self.camid = camid
        self.frame_infos = frame_infos

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __getitem__(self, idx):
        seq, frame = self.frame_infos[idx]
        frame_path = join(self.indir, self.modality,
                          self.camid, str(seq), f"{frame}.jpg")
        out_label_path = join(
            self.outdir_dict["labels"], self.modality, self.camid, str(seq), f'{frame}.npy')
        out_img_path = join(
            self.outdir_dict["images"], self.modality, self.camid, str(seq), f'{frame}.jpg')

        img = Image.open(frame_path).convert('RGB')
        img = self.transform(img)

        # "label_path": out_label_path,
        return {"index": idx, "img": img, "in_img_path": frame_path, "out_img_path": out_img_path, "out_label_path": out_label_path}

    def __len__(self):
        return len(self.frame_infos)


def postprocess_automatic_mask(inputs, outputs, device='cpu'):
    """
    Inputs:
    inputs - (dictionary) consists of keys 'index', 'img', 'img_path'
    outputs - (dictionary) consists of keys 'img', 'mask'
    """
    out_img_paths = inputs["out_img_path"]
    out_label_paths = inputs["out_label_path"]

    for data_idx, anns in enumerate(outputs):
        # SAM2
        # sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        # masks = [ann['segmentation'] for ann in sorted_anns]
        masks = anns['masks']
        img = inputs['img'][data_idx][None, ...].to(device)
        masks = torch.from_numpy(np.array(masks)).unsqueeze(
            0).to(device)  # [1, C, H, W]

        labels = filter_segment_mask(masks)
        # [B, HW] Set 0 to unlabeled
        # labels = torch.argmax(masks.long(), dim=1)

        # 3 Make labels contiguous
        labels = make_labels_contiguous_vectorized(labels).to(torch.uint16)

        # 4 Make labels contiguous
        blended_img = vis.show_masks_on_image(img, labels)

        # 5 Save labels and images (save labels as uint8)
        labels = labels.squeeze(0).cpu().numpy()
        # labels.tofile(out_label_paths[data_idx])
        np.save(out_label_paths[data_idx], labels)
        cv2.imwrite(out_img_paths[data_idx], blended_img)
        print(f"Saved {out_label_paths[data_idx]}")


def process_sam_chunk(inputs):
    """
    Process a chunk of image filepaths to save SAM masks for
    """
    frames_infos, indir, outdir_dict, sensor_dict, mask_type, device = inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mask_type == "dynamic":
        video_subdir = join(
            sensor_dict[0][0], sensor_dict[0][1], str(frames_infos[0][0]))
        video_dir = join(outdir_dict["raw"], video_subdir)
        step = 1
        batch_size = 1
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Initialize SAM2 models
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = "./external/sam2/checkpoints/sam2.1_hiera_large.pt"
        video_predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(
            model_cfg, sam2_checkpoint, device=device)

        image_predictor = SAM2ImagePredictor(sam2_image_model)

        inference_state = video_predictor.init_state(
            video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)

        # Init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(device)
        # Iterate through dictionary keys and construct string for grounding
        text = [f"{key}." for key, val in SAM_DYNAMIC_LABEL_MAP.items() if val!=0]
        text = " ".join(text)
        sam2_masks = MaskDictionaryModel()
        PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
        objects_count = 0

        output_video_path = join(
            outdir_dict["images"], f"{frames_infos[0][0]}.mp4")

        mask_data_dir = os.path.join(
            outdir_dict["labels"], video_subdir)
        json_data_dir = os.path.join(outdir_dict["json"], video_subdir)
        result_dir = os.path.join(outdir_dict["images"], video_subdir)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(mask_data_dir, exist_ok=True)
        os.makedirs(json_data_dir, exist_ok=True)
    else:
        mask_batch_size = 1
        batch_size = 1
        mask_generator = pipeline(
            "mask-generation", model="facebook/sam-vit-huge", batch_size=mask_batch_size, device=device)
        # mask_generator = SAM2AutomaticMaskGenerator(
        #     model=sam2_image_model,  # Your pre-trained SAM2 model
        #     points_per_side=64,  # Increase to make the grid denser
        #     crop_n_layers=1,  # Enable cropping to further cover areas
        #     crop_overlap_ratio=0.3,  # Adjust overlap to avoid filtering too aggressively
        #     min_mask_region_area=20,  # Keep all mask regions
        #     multimask_output=True,  # Generate multiple masks per point
        #     pred_iou_thresh=0.6,  # Lower IoU threshold to retain more masks
        #     stability_score_thresh=0.85,  # Lower stability threshold to retain more masks
        #     mask_threshold=0.1,  # Adjust mask threshold to keep valid masks
        #     box_nms_thresh=0.9,  # Allow more overlapping masks through NMS
        # )

    # 2 Initialize and propagate mask tracker
    for sensor in sensor_dict:
        modality, camid = sensor
        dataset = SequentialDataset(
            outdir_dict, modality, camid, frames_infos)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=12)
        frame_names = [f'{p[1]}.jpg' for p in dataset.frame_infos]
        # Check that frame_names are contiguous

        for inputs in dataloader:
            # prompt grounding dino to get the box coordinates on specific frame
            start_frame_idx = inputs["index"][0].item()
            # if start_frame_idx < 3380:
            #     continue

            img = inputs["img"].to(device)
            print("start_frame_idx", start_frame_idx)

            # img_path = os.path.join(video_dir, frame_names[start_frame_idx])

            """
            Step 6: Save the automatically generated sam masks
            """
            if mask_type == "static":
                frame_paths = inputs["in_img_path"]
                # Convert tensor of images to PIL images
                images = [F.to_pil_image(img_th) for img_th in img]
                outputs = mask_generator(
                    images, points_per_side=32, points_per_batch=1024)
                # outputs = mask_generator(frame_paths,
                #                          points_per_side=128,
                #                          points_per_batch=32,
                #                          pred_iou_threshold=0.95,
                #                          box_nms_thresh=0.8)
                # outputs = []
                # for img_th in img:
                #     img_np = img_th.permute(1, 2, 0).cpu().numpy()
                #     output = mask_generator.generate(img_np)
                #     outputs.append(output)

                postprocess_automatic_mask(inputs, outputs, device=device)
            else:
                assert batch_size == 1, "Batch size must be 1 for manual mask generation"
                img_path = inputs["in_img_path"][0]
                image_base_name = os.path.basename(img_path).split(".")[0]
                image = Image.open(img_path)
                # run Grounding DINO on the image
                inputs = processor(images=image, text=text,
                                   return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

                mask_dict = MaskDictionaryModel(
                    promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.25,
                    text_threshold=0.25,
                    target_sizes=[image.size[::-1]]
                )

                # prompt SAM image predictor to get the mask for the object
                image_predictor.set_image(np.array(image.convert("RGB")))

                # process the detection results
                input_boxes = results[0]["boxes"]  # .cpu().numpy()
                # print("results[0]",results[0])
                OBJECTS = results[0]["labels"]
                if len(input_boxes) == 0:
                    print("No object detected in the frame, skip the frame {}".format(
                        start_frame_idx))
                    continue

                # Prompt SAM 2 image predictor to get the mask for the object
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                # convert the mask shape to (n, H, W)
                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)

                """
                Step 3: Register each object's positive points to video predictor
                """

                # If you are using point prompts, we uniformly sample positive points based on the mask
                if mask_dict.promote_type == "mask":
                    mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(
                        device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
                else:
                    raise NotImplementedError(
                        "SAM 2 video predictor only support mask prompts")

                """
                Step 4: Propagate the video predictor to get the segmentation results for each frame
                """
                objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
                print("objects_count", objects_count)
                video_predictor.reset_state(inference_state)
                if len(mask_dict.labels) == 0:
                    print("No object detected in the frame, skip the frame {}".format(
                        start_frame_idx))
                    continue
                video_predictor.reset_state(inference_state)

                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )

                video_segments = {}  # output the following {step} frames tracking masks
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                    frame_masks = MaskDictionaryModel()

                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0)  # .cpu().numpy()
                        object_info = ObjectInfo(
                            instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id))
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info
                        if out_frame_idx > len(frame_names) - 1:
                            print("Invalid out frame idx", out_frame_idx)
                            import pdb
                            pdb.set_trace()
                            continue
                        image_base_name = frame_names[out_frame_idx].split(".")[
                            0]
                        frame_masks.mask_name = f"mask_{image_base_name}.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]

                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = copy.deepcopy(frame_masks)

                print("video_segments:", len(video_segments))
                """
                Step 5: save the tracking masks and json files
                """
                for frame_idx, frame_masks_info in video_segments.items():
                    mask = frame_masks_info.labels
                    mask_img = torch.zeros(frame_masks_info.mask_height,
                                           frame_masks_info.mask_width)
                    for obj_id, obj_info in mask.items():
                        mask_img[obj_info.mask == True] = obj_id

                    mask_img = mask_img.numpy().astype(np.uint16)
                    np.save(os.path.join(mask_data_dir,
                            frame_masks_info.mask_name), mask_img)

                    json_data = frame_masks_info.to_dict()
                    json_data_path = os.path.join(
                        json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                    with open(json_data_path, "w") as f:
                        json.dump(json_data, f)
        if mask_type == "static":
            pass
        else:
            """
            Step 6: Draw the results and save the video
            """
            CommonUtils.draw_masks_and_box_with_supervision(
                video_dir, mask_data_dir, json_data_dir, result_dir)

            create_video_from_images(
                result_dir, output_video_path, frame_rate=10)


def preprocess_sam2_chunk(inputs):
    """
    Process a chunk of image filepaths to save SAM masks for
    """
    frame_infos, indir, outdir_dict, sensor_dict, mask_type, device = inputs

    # outimg_dir = join(outdir_dict["raw"], sensor_dict[0]
    #                   [0], sensor_dict[0][1], str(frame_infos[fidx][0]))

    # if not os.path.exists(outimg_dir):
    #     os.makedirs(outimg_dir)
    #     print("Created output directory: ", outimg_dir)
    # else:
    #     if len(os.listdir(outimg_dir)) == len(frame_infos):
    #         print("Already preprocessed: ", outimg_dir)
    #         return

    for fidx in tqdm(range(len(frame_infos))):
        inimg_dir = join(indir, sensor_dict[0][0],
                         sensor_dict[0][1], str(frame_infos[fidx][0]))
        outimg_dir = join(outdir_dict["raw"], sensor_dict[0]
                          [0], sensor_dict[0][1], str(frame_infos[fidx][0]))
        if not os.path.exists(outimg_dir):
            os.makedirs(outimg_dir)
            print("Created output directory: ", outimg_dir)
        output_file = f"{frame_infos[fidx][1]}.jpg"
        output_filename = output_file.split("_")[-1]
        output_path = join(outimg_dir, output_file)
        if os.path.exists(output_path):
            print("Already preprocessed: ", output_path)
            continue

        img_file = frame2fn(
            sensor_dict[0][0],
            sensor_dict[0][1],
            frame_infos[fidx][0],
            frame_infos[fidx][1],
            "png")

        img_path = join(inimg_dir, img_file)
        if not os.path.exists(img_path):
            img_path = img_path.replace("png", "jpg")
        img = cv2.imread(img_path)
        cv2.imwrite(output_path, img)


def main(args):
    """
    Setsup main queue of models and chunks to process
    """
    indir = args.indir
    outdir = args.outdir
    camids = args.camids
    skip_factor = args.skip_factor
    img_ds = args.img_ds
    skip_sequences = args.skip_sequences
    NUM_CHUNKS = args.num_chunks
    CHUNK_IDX = args.chunk_idx

    # 1 Load all frames
    sensor_dict = []
    for camid in camids:
        sensor_dict.append([CAMERA_DIR, camid])

    input_frames_dict = get_frames_from_json(
        indir, sensor_dict, override_all=True, ds=skip_factor)

    # 2 Aggregate frames into unified infos and exlude sequences
    unified_infos = np.empty((0, 2), dtype=int)
    frame_dividers = []
    for seq in input_frames_dict.keys():
        if seq in skip_sequences:
            continue
        for camid in input_frames_dict[seq].keys():
            seq_infos = np.ones_like(
                input_frames_dict[seq][camid], dtype=int) * seq
            frame_infos = np.array(
                [fn2frame(frame_path) for frame_path in input_frames_dict[seq][camid]], dtype=int
            )

            infos = np.stack((seq_infos, frame_infos), axis=1)
            unified_infos = np.vstack((unified_infos, infos))
            frame_dividers.append(len(infos))

    print("LOADED FRAMES FOR SEQUENCES\n", np.unique(unified_infos[:, 0]))
    if args.mask_type == "dynamic":
        frame_dividers = np.array(frame_dividers)
        frame_chunks = np.split(unified_infos, np.cumsum(frame_dividers)[:-1])
    else:
        frame_dividers = np.array(frame_dividers)
        frame_chunks = np.split(unified_infos, np.cumsum(frame_dividers)[:-1])
        # frame_chunks = np.array_split(unified_infos, NUM_CHUNKS)

    if args.mask_type == "static":
        label_dir = join(outdir, "static")
        os.makedirs(label_dir, exist_ok=True)
        image_dir = join(outdir, "static_images")
    else:
        label_dir = join(outdir, "dynamic")
        os.makedirs(label_dir, exist_ok=True)
        image_dir = join(outdir, "dynamic_images")

    # Create outdir for all cams and sequences
    outdir_dict = {
        "labels": label_dir,
        "images": image_dir,
        "raw": join(outdir, "raw"),
        "json": join(outdir, "json"),
    }
    for key, keydir in outdir_dict.items():
        for camid in camids:
            for seq in np.unique(unified_infos[:, 0]):
                os.makedirs(join(keydir, CAMERA_DIR, str(
                    camid), str(seq)), exist_ok=True)

    # 3 Create workers and process chunks with SAM
    indir_list = [indir] * len(frame_chunks)
    outdir_list = [outdir_dict] * len(frame_chunks)
    sensor_dict_list = [sensor_dict] * len(frame_chunks)
    mask_type_list = [args.mask_type] * len(frame_chunks)

    num_devices = torch.cuda.device_count()
    print("NUM DEVICES: ", num_devices)
    cuda_device_list = [f'cuda:{i}' for i in range(num_devices)]
    cuda_device_args = np.tile(
        np.array(cuda_device_list), len(frame_chunks)//num_devices + 1)

    task_args = [
        (frame_chunks[i], indir_list[i], outdir_list[i], sensor_dict_list[i], mask_type_list[i], cuda_device_args[i]) for i in range(len(frame_chunks))
    ]

    preprocess_sam2_chunk(task_args[CHUNK_IDX])

    print("PROCESSING SAM CHUNK ", CHUNK_IDX)
    process_sam_chunk(task_args[CHUNK_IDX])

    # TODO: Figure out how get mp to work with spawned threads. Doesn't play nicely.
    # for task_arg in task_args:
    #     process_sam_chunk(task_arg)

    # process_sam_chunk(task_args[0])
    # import pdb; pdb.set_trace()
    # with Pool(processes=num_devices) as pool:
    #     with tqdm(total=len(task_args)) as pbar:
    #         for _ in pool.imap(process_sam_chunk, task_args):
    #             pbar.update(1)


if __name__ == '__main__':
    # mp.set_start_method('spawn')  # Set the start method for multiprocessing
    args = parse_args()
    main(args)
