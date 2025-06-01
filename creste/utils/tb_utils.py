
import os
import pickle 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from creste.utils.feature_extractor import get_robust_pca

import matplotlib.pyplot as plt

def make_label_sprites(labels, color_map, patch_size=16):
    """
    Create image sprites for tensorboard visualization. Use label idx to map to color, and then
    create a sprite of the images where each sprite is the solid color corresponding to the label

    Inputs:
        labels - torch.Tensor, labels for each image
        color_map - list [Nx3], mapping from label idx to color
    Returns:
        image_patches - torch.Tensor, image patches for labels [N x 3 x patch_size x patch_size]
    """
    color_map = torch.tensor(color_map, dtype=torch.float32)
    color_map[:, [0,1,2]] = color_map[:, [2,1,0]] # Convert from BGR to RGB
    
    # Create a patch_size x patch_size tensor for each label using the colormap
    image_patches = torch.zeros((len(labels), 3, patch_size, patch_size), dtype=torch.float32)
    for i in range(len(color_map)):
        mask = labels == i
        patch = torch.ones((3, patch_size, patch_size), dtype=torch.float32)
        image_patches[mask] = patch * color_map[i].reshape(3, 1, 1) / 255.0
    return image_patches

def save_embeddings(feats, labels, embed_dict, verbose=False):
    """
    Save embeddings and labels to disk
    Inputs:
        feats: (torch.Tensor) [N, F] tensor of embeddings
        labels: (torch.Tensor) [N] tensor of labels
    """
    if verbose:
        print(f'Saving embeddings to {embed_dict["feats"]}, and labels to {embed_dict["labels"]}')
    os.makedirs(os.path.dirname(embed_dict["feats"]), exist_ok=True)
    os.makedirs(os.path.dirname(embed_dict["labels"]), exist_ok=True)

    with open(embed_dict["feats"], 'wb') as f:
        pickle.dump(feats, f)
    with open(embed_dict["labels"], 'wb') as f:
        pickle.dump(labels, f)
    
def load_embeddings(embed_dict):
    """
    Input:
        embed_dict: (dict) dictionary containing the path to the embeddings and labels
            "feats": str, path to the embeddings
            "labels": str, path to the labels
    Output:
        feats: (np.ndarray) [N, F] array of embeddings
        labels: (np.ndarray) [N,] array of labels
    """
    assert os.path.exists(embed_dict["feats"]), f"Embedding file not found at {embed_dict['feats']}"
    assert os.path.exists(embed_dict["labels"]), f"Label file not found at {embed_dict['labels']}"
    print(f'Loading embeddings from {embed_dict["feats"]} and labels from {embed_dict["labels"]}')

    with open(embed_dict["feats"], 'rb') as f:
        feats = pickle.load(f)
    with open(embed_dict["labels"], 'rb') as f:
        labels = pickle.load(f)
    
    return feats, labels

def add_embedding_to_tb(writer, feats, labels, label_to_name, label_to_color, tag, global_step):
    """
    sem_feats: (np.ndarray) [N, F] np array of embeddings
    sem_labels: (np.ndarray) [N] np array of labels
    label_to_name: (list) [C] np array of class names
    label_to_color: (list) [C, 3] np array of colors
    """
    sem_sprites = None
    if label_to_color is not None:
        sem_sprites = make_label_sprites(labels, label_to_color)

    label_names = np.array(label_to_name, dtype=str)[labels]
    writer.add_embedding(feats, label_img=sem_sprites, metadata=label_names, tag=tag, global_step=global_step)

def write_to_tb(sem_feats, sem_labels, sem_label_class_names, embed_dict, sem_id_to_color=None, verbose=False):
    """
    Inputs:
        sem_feats: (np.ndarray) [N, F] tensor of embeddings
        sem_labels: (np.ndarray) [N] tensor of labels
    """
    sem_sprites = None
    if sem_id_to_color is not None:
        sem_sprites = make_label_sprites(sem_labels, sem_id_to_color)

    # Save embeddings
    emb_dir = os.path.dirname(embed_dict["feats"])
    os.makedirs(emb_dir, exist_ok=True)
    save_embeddings(sem_feats, sem_labels, embed_dict, verbose=verbose)

    #4.2 Save latent features to tensorboard
    tensor_dir = embed_dict["tb"]
    label2class = np.array(sem_label_class_names, dtype=str)
    sem_labels = label2class[sem_labels]
    num_classes = len(sem_label_class_names)
    
    if verbose:
        print(f'Saved feats to tensorboard here {tensor_dir}')
    writer = SummaryWriter(tensor_dir)
    writer.add_embedding(sem_feats, label_img=sem_sprites, metadata=sem_labels, tag=f'sem_feats_{num_classes}class')
    writer.close()

def log_depth_img_to_tb(tb_writer, inputs, outputs, keys, epoch, global_step, prefix='val'):
    """
    Logs predicted depth images to tensorboard
    """
    for key in keys:
        if key in outputs:
            depth_img = outputs[key]
        elif key in inputs:
            depth_img = inputs[key]
        else:
            print(f"Key {key} not found in output")
            continue

        if "depth" not in key:
            continue

        # Normalize depth image and convert to grayscale
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        tb_writer.experiment.add_image(f'{prefix}/{key}/{epoch}', make_grid(depth_img.unsqueeze(1), nrow=8), global_step=global_step)


def log_feat_img_to_tb(tb_writer, inputs, outputs, keys, epoch, global_step, prefix='val'):
    """
    Indexes into output using keys to save images to tensorboard

    Inputs:
        output: (dict) dictionary of outputs
        keys: (list) list of keys to index into output
        tb_dir: (str) directory to save tensorboard images
        step: (int) training step
    """
    for key in keys:
        if key in outputs:
            feat_img = outputs[key]
        elif key in inputs:
            feat_img = inputs[key]
        else:
            print(f"Key {key} not found in output")
            continue

        if "depth" in key:
            continue # log depth elsewhere

        # PCA reduce feature dim
        with torch.no_grad():
            if feat_img.dim() == 5:
                B, Vp1, F, H, W = feat_img.shape
                feat_img = feat_img.permute(0, 1, 3, 4, 2).reshape(-1, F)
            elif feat_img.dim() == 4:
                B, F, H, W = feat_img.shape
                Vp1 = 1
                feat_img = feat_img.permute(0, 2, 3, 1).reshape(-1, F)

            reduction_mat, rgb_feat_min, rgb_feat_max = get_robust_pca(feat_img)
            feat_img = feat_img @ reduction_mat
            feat_img = (feat_img - rgb_feat_min) / (rgb_feat_max - rgb_feat_min)
            feat_img = feat_img.reshape(B*Vp1, H, W, -1).permute(0, 3, 1, 2) # [B, V+1, H, W, C] -> [B, V+1, C, H, W]
            tb_writer.experiment.add_image(f'{prefix}/{key}/{epoch}', make_grid(feat_img), global_step=global_step)
        
def add_plot(tb_writer, plot_name, global_step):
    """
    Add a plot to tensorboard
    """
    tb_writer.experiment.add_figure(plot_name, plt.gcf(), global_step=global_step)
    plt.close()
    