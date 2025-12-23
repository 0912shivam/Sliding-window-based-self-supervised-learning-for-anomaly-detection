import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
import json
from sklearn.neighbors import KernelDensity
import time
from tqdm import tqdm

def evaluate_image(args, model, train_loader, test_loader, device, category='dbt'):
    model.eval()

    # Extract Normal Image Features
    embedding_list = []
    patch_size = args.patch_size
    sliding_step = args.step_size

    patch_embeddings = None

    print(f'\n[Evaluation] Extracting features from {len(train_loader)} training images...')
    with torch.no_grad():
        for idx, data in enumerate(tqdm(train_loader, desc='Extracting train features', unit='batch')):
            curr = time.time()
            # Fit img to size B * 3 * H * W
            img, _ = data
            img = img.to(device)

            x_length = (img.shape[2] - patch_size) // sliding_step
            y_length = (img.shape[3] - patch_size) // sliding_step

            single_patch_embeddings = None
            for i in range(x_length):
                for j in range(y_length):
                    start_x = i*sliding_step
                    start_y = j*sliding_step
                    curr_patch = img[:,:,start_x:start_x+patch_size,start_y:start_y+patch_size]
                    _, feature = model(curr_patch)
                    feature = feature.cpu().squeeze().unsqueeze(-1)
                    if single_patch_embeddings is None:
                        single_patch_embeddings = feature
                    else:
                        single_patch_embeddings = torch.cat([single_patch_embeddings, feature], dim=-1)

            batch_size = single_patch_embeddings.shape[0]
            hidden_size = single_patch_embeddings.shape[1]
            single_patch_embeddings = single_patch_embeddings.reshape(batch_size,hidden_size,-1).cpu()

            if patch_embeddings is None:
                patch_embeddings = single_patch_embeddings
            else:
                patch_embeddings = torch.cat([patch_embeddings, single_patch_embeddings], dim=0)

    print(f'[Evaluation] Patch embedding size: {patch_embeddings.shape}')
    _, patch_dim, patch_num = patch_embeddings.shape

    # Testing
    gt_list_img_lvl = []
    pred_list_img_lvl = []
    score_patch_list = []
    
    patch_embeddings = patch_embeddings.numpy()
    print(f'[Evaluation] Testing on {len(test_loader)} images...')
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc='Testing', unit='img')):
            curr = time.time()
            img, label = data

            img = img.to(device)
            embedding_tests = None
            for i in range(x_length):
                for j in range(y_length):
                    start_x = i*sliding_step
                    start_y = j*sliding_step
                    curr_patch = img[:,:,start_x:start_x+patch_size,start_y:start_y+patch_size]
                    _, feature = model(curr_patch)
                    feature = feature.cpu().squeeze().unsqueeze(-1)
                    if embedding_tests is None:
                        embedding_tests = feature
                    else:
                        embedding_tests = torch.cat([embedding_tests, feature], dim=-1)
            


            # hidden * size
            embedding_tests = embedding_tests.squeeze().cpu()
            embedding_tests = embedding_tests.reshape(hidden_size, -1).numpy()
            
            # Compute distance with double batching to avoid memory issues
            # Process both train and test patches in small chunks
            num_test_patches = embedding_tests.shape[1]
            num_train_patches = patch_embeddings.shape[0]
            test_chunk_size = 20  # Process 20 test patches at a time
            train_chunk_size = 100  # Process 100 train patches at a time
            score_patches = []
            
            for test_start in range(0, num_test_patches, test_chunk_size):
                test_end = min(test_start + test_chunk_size, num_test_patches)
                test_chunk = embedding_tests[:, test_start:test_end]  # Shape: (2048, 20)
                
                min_distances = []
                for train_start in range(0, num_train_patches, train_chunk_size):
                    train_end = min(train_start + train_chunk_size, num_train_patches)
                    train_chunk = patch_embeddings[train_start:train_end, :, 0]  # Shape: (100, 2048)
                    
                    # Compute distances between train_chunk and test_chunk
                    # train_chunk: (100, 2048), test_chunk: (2048, 20)
                    dis = np.linalg.norm(
                        train_chunk[:, :, np.newaxis] - test_chunk[np.newaxis, :, :],
                        axis=1
                    )  # Shape: (100, 20)
                    min_distances.append(np.min(dis, axis=0))  # Min over train patches
                
                # Get overall minimum distance for this test chunk
                min_dist_chunk = np.min(np.array(min_distances), axis=0)
                score_patches.extend(min_dist_chunk.tolist())
            
            score_patches = np.array(score_patches)
            image_score = max(score_patches)

            pred_list_img_lvl.append(float(image_score))
            gt_list_img_lvl.append(label.numpy()[0])
            score_patch_list.append(score_patches.tolist())

        pred_img_np = np.array(pred_list_img_lvl)
        gt_img_np = np.array(gt_list_img_lvl)
        img_auc = roc_auc_score(gt_img_np, pred_img_np)
        print(f'\n[Evaluation Complete] Image-level AUC-ROC: {img_auc:.4f}')

        file_name = 'results/performance_%s_%s.json' % (category, str(img_auc))
        with open(file_name, 'w+') as f:
            gt_list_int = [int(i) for i in gt_list_img_lvl]
            json.dump([gt_list_int, pred_list_img_lvl, score_patch_list], f)

    model.train()
    return img_auc
