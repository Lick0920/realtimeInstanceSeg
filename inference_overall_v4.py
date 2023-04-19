# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import copy
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from sparseinst import VisualizationDemo, add_sparse_inst_config
import numpy as np
import json
# # this part facilitates correct packaging
# def script_method(fn, _rcb=None):
#     return fn
# def script(obj, optimize=True, _frames_up=0, _rcb=None):
#     return obj
# import torch.jit
# script_method1 = torch.jit.script_method
# script1 = torch.jit.script
# torch.jit.script_method = script_method
# torch.jit.script = script
# # end of this part
import torch
import ot
torch.set_grad_enabled(False)
torch.cuda.set_device(0)
import torch.nn.functional as F
# constants
WINDOW_NAME = "COCO detections"

dump_middle_result = 0
instance_thresh = 0.7
inference_batch_size = 32
detection_thresh = 0.25
hair_cls = 1
maximum_possible_number = 9999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combine_hair_thresh = 0.1
img_patch_size = 512
GPU_utilization = 0.5
update_tracking_hair_thresh = 0.4
FA_filter_thresh = 0.5
filter_FA_overlap_with_root_thresh = 0.8
assert((GPU_utilization >= 0 and GPU_utilization <= 1))
inter_patch_step = int(int(img_patch_size / 4 * 3) * GPU_utilization + int(img_patch_size / 8 * 7) * (1.0 - GPU_utilization))
########################################################################################################################################
##################################### functions ########################################################################################
########################################################################################################################################
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs\\sparse_inst_r50vd_dcn_giam_aug.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def compute_hori_center(img_patch):
    contours, _ = cv2.findContours(img_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hori_center, vert_center = 0, 0
    for idx in range(len(contours)):
        if cv2.moments(contours[idx])['m00'] == 0:
            return
        hori_center += int(cv2.moments(contours[idx])['m10'] / cv2.moments(contours[idx])['m00'])
        vert_center += int(cv2.moments(contours[idx])['m01'] / cv2.moments(contours[idx])['m00'])
    hori_center = hori_center / len(contours)
    return hori_center

def compute_hori_center_gpu(img_patch):
    if torch.sum(img_patch) == 0:
        return
    img_patch_H, img_patch_W = img_patch.shape[0], img_patch.shape[1]
    hori_coord_meshgrid = torch.meshgrid(torch.range(0, img_patch_H - 1), torch.range(0, img_patch_W - 1))[1].to(device)
    return (torch.sum(torch.mul(hori_coord_meshgrid, img_patch)) / torch.where(img_patch)[0].shape[0]).item()

def compute_iou_single_box(curr_img_boxes, next_img_boxes):# Order: top, bottom, left, right
    intersect_vert = min([curr_img_boxes[1], next_img_boxes[1]]) - max([curr_img_boxes[0], next_img_boxes[0]])
    intersect_hori = min([curr_img_boxes[3], next_img_boxes[3]]) - max([curr_img_boxes[2], next_img_boxes[2]])
    union_vert = max([curr_img_boxes[1], next_img_boxes[1]]) - min([curr_img_boxes[0], next_img_boxes[0]])
    union_hori = max([curr_img_boxes[3], next_img_boxes[3]]) - min([curr_img_boxes[2], next_img_boxes[2]])
    if intersect_vert > 0 and intersect_hori > 0 and union_vert > 0 and union_hori > 0:
        corresponding_coefficient = float(intersect_vert) * float(intersect_hori) / (float(curr_img_boxes[1] - curr_img_boxes[0]) * float(curr_img_boxes[3] - curr_img_boxes[2]) + float(next_img_boxes[1] - next_img_boxes[0]) * float(next_img_boxes[3] - next_img_boxes[2]) - float(intersect_vert) * float(intersect_hori))
    else:
        corresponding_coefficient = 0.0
    return corresponding_coefficient

def ot_iou(prev_mask, curr_mask, mask_height, mask_width):
    prev_mask_ele1_min, prev_mask_ele1_max = torch.min(prev_mask[1]), torch.max(prev_mask[1])
    prev_mask_ele2_min, prev_mask_ele2_max = torch.min(prev_mask[2]), torch.max(prev_mask[2])
    curr_mask_ele1_min, curr_mask_ele1_max = torch.min(curr_mask[0]), torch.max(curr_mask[0])
    curr_mask_ele2_min, curr_mask_ele2_max = torch.min(curr_mask[1]), torch.max(curr_mask[1])
    if prev_mask_ele1_max <= curr_mask_ele1_min or prev_mask_ele1_min >= curr_mask_ele1_max or \
            prev_mask_ele2_max <= curr_mask_ele2_min or prev_mask_ele2_min >= curr_mask_ele2_max:
        return 0

    ot_iou_start = time.time()
    # prev_list = torch.cat([prev_mask[1], prev_mask[2]], axis=0).reshape((2, prev_mask[1].shape[0]))
    # curr_list = torch.cat([curr_mask[0], curr_mask[1]], axis=0).reshape((2, curr_mask[0].shape[0]))
    # prev_list_list = torch.transpose(prev_list, 0, 1).tolist()
    # curr_list_list = torch.transpose(curr_list, 0, 1).tolist()
    # common_num = len([x for x in prev_list_list if x in curr_list_list])

    min_prev_curr_vert, min_prev_curr_hori = min([torch.min(prev_mask[1]).item(), torch.min(curr_mask[0]).item()]), min([torch.min(prev_mask[2]).item(), torch.min(curr_mask[1]).item()])
    max_prev_curr_vert, max_prev_curr_hori = max([torch.max(prev_mask[1]).item(), torch.max(curr_mask[0]).item()]), max([torch.max(prev_mask[2]).item(), torch.max(curr_mask[1]).item()])
    seek_common_ele = torch.zeros((max_prev_curr_vert - min_prev_curr_vert + 1, max_prev_curr_hori - min_prev_curr_hori + 1)).byte().to(device)
    ot_iou_middle = time.time()
    seek_common_ele[prev_mask[1] - min_prev_curr_vert, prev_mask[2] - min_prev_curr_hori] = 1
    seek_common_ele[curr_mask[0] - min_prev_curr_vert, curr_mask[1] - min_prev_curr_hori] = seek_common_ele[curr_mask[0] - min_prev_curr_vert, curr_mask[1] - min_prev_curr_hori] + 1
    common_num = len(torch.where(seek_common_ele == 2)[0])
    # del seek_common_ele
    # torch.cuda.empty_cache()

    # seek_common_ele = np.zeros((mask_height, mask_width))
    # ot_iou_middle = time.time()
    # seek_common_ele[prev_mask[1], prev_mask[2]] = 1
    # seek_common_ele[curr_mask[0], curr_mask[1]] = seek_common_ele[curr_mask[0], curr_mask[1]] + 1
    # common_num = len(np.where(seek_common_ele == 2)[0])

    ot_iou_end = time.time()
    return common_num / (common_num + min([prev_mask[1].shape[0] - common_num, curr_mask[0].shape[0] - common_num]))#(common_num + prev_distinct_num + curr_distinct_num)

def ot_matching_matrix(matching_matrix):
    if matching_matrix.shape[0] > matching_matrix.shape[1]:
        add_width = matching_matrix.shape[0] - matching_matrix.shape[1]
        matching_matrix = np.concatenate((matching_matrix, np.ones((matching_matrix.shape[0], add_width)) * np.max(matching_matrix) * 2), axis=1)
        ot_src = np.array([1.0] * matching_matrix.shape[0])
        ot_dst = np.array([1.0] * matching_matrix.shape[1])
        prev_curr_transportation_array = ot.emd(ot_src, ot_dst, matching_matrix)
        prev_curr_transportation_array = prev_curr_transportation_array[:, :prev_curr_transportation_array.shape[1] - add_width]
    elif matching_matrix.shape[0] < matching_matrix.shape[1]:
        add_height = matching_matrix.shape[1] - matching_matrix.shape[0]
        matching_matrix = np.concatenate((matching_matrix, np.ones((add_height, matching_matrix.shape[1])) * np.max(matching_matrix) * 2), axis=0)
        ot_src = np.array([1.0] * matching_matrix.shape[0])
        ot_dst = np.array([1.0] * matching_matrix.shape[1])
        prev_curr_transportation_array = ot.emd(ot_src, ot_dst, matching_matrix)
        prev_curr_transportation_array = prev_curr_transportation_array[:prev_curr_transportation_array.shape[0] - add_height, :]
    else:
        ot_src = np.array([1.0] * matching_matrix.shape[0])
        ot_dst = np.array([1.0] * matching_matrix.shape[1])
        prev_curr_transportation_array = ot.emd(ot_src, ot_dst, matching_matrix)
    return prev_curr_transportation_array

# Point 1: fit a line for each instance in each patch, determine the same instance from different patches by slope
def prev_curr_hairs_matching(prev_patch_hair_mask_list, curr_patch_hair_mask_list, curr_patch_top, curr_patch_left, img_patch_size, inter_patch_step):
    prev_curr_hairs_matching_matrix = np.zeros((len(prev_patch_hair_mask_list), len(curr_patch_hair_mask_list))).astype('float32')
    for prev_item_idx in range(len(prev_patch_hair_mask_list)):
        for curr_item_idx in range(len(curr_patch_hair_mask_list)):
            time_before_matching = time.time()
            prev_item = prev_patch_hair_mask_list[prev_item_idx]
            curr_item = curr_patch_hair_mask_list[curr_item_idx]
            time_before_matching2 = time.time()
            curr_vert_coords, curr_hori_coords = torch.where(curr_item[curr_patch_top:curr_patch_top+img_patch_size, curr_patch_left:curr_patch_left+img_patch_size] > 0)[0] + curr_patch_top, \
                                                 torch.where(curr_item[curr_patch_top:curr_patch_top+img_patch_size, curr_patch_left:curr_patch_left+img_patch_size] > 0)[1] + curr_patch_left # torch.where(curr_item > 0)[0], torch.where(curr_item > 0)[1]
            prev_vert_coords, prev_hori_coords = torch.where(prev_item > 0)[0], torch.where(prev_item > 0)[1] # torch.where(prev_item > 0)[0], torch.where(prev_item > 0)[1]
            time_before_matching3 = time.time()
            if min([curr_hori_coords.shape[0], curr_vert_coords.shape[0], prev_vert_coords.shape[0], prev_hori_coords.shape[0]]) == 0:
                prev_curr_hairs_matching_matrix[prev_item_idx, curr_item_idx] = maximum_possible_number
                continue
            curr_left, curr_top, curr_right, curr_bottom = torch.min(curr_hori_coords), torch.min(curr_vert_coords), torch.max(curr_hori_coords), torch.max(curr_vert_coords)
            prev_left, prev_top, prev_right, prev_bottom = torch.min(prev_hori_coords), torch.min(prev_vert_coords), torch.max(prev_hori_coords), torch.max(prev_vert_coords)
            if compute_iou_single_box([curr_top, curr_bottom, curr_left, curr_right], [prev_top, prev_bottom, prev_left, prev_right]) <= 0:
                prev_curr_hairs_matching_matrix[prev_item_idx, curr_item_idx] = maximum_possible_number
                continue
            time_before_fitline = time.time()
            prev_output = cv2.fitLine(torch.cat((prev_vert_coords.reshape(prev_vert_coords.shape[0], 1), prev_hori_coords.reshape(prev_hori_coords.shape[0], 1)), 1).cpu().numpy(), cv2.DIST_L2, 0, 0.01, 0.01)
            time_after_fitline = time.time()
            # print('fit line time: ' + str(time_after_fitline - time_before_fitline))
            prev_k = prev_output[1] / prev_output[0]
            # prev_b = prev_output[3] - prev_k * prev_output[2]
            curr_output = cv2.fitLine(torch.cat((curr_vert_coords.reshape(curr_vert_coords.shape[0], 1), curr_hori_coords.reshape(curr_hori_coords.shape[0], 1)), 1).cpu().numpy(), cv2.DIST_L2, 0, 0.01, 0.01)
            curr_k = curr_output[1] / curr_output[0]
            time_after_matching = time.time()
            prev_curr_hairs_matching_matrix[prev_item_idx, curr_item_idx] = abs(prev_k[0] - curr_k[0])

            del prev_item, curr_item, curr_vert_coords, curr_hori_coords, prev_vert_coords, prev_hori_coords, curr_left, curr_top, curr_right, curr_bottom, prev_left, prev_top, prev_right, prev_bottom,
            torch.cuda.empty_cache()
    return prev_curr_hairs_matching_matrix

########################################################################################################################################
##################################### main flow ########################################################################################
########################################################################################################################################
# step 1: downsample images for root segmentation
input_dirs = json.load(open('setting.json', 'r'))
src_dir = input_dirs['src_dir']
dst_dir = input_dirs['dst_dir']
separate_dir = input_dirs['separate_dir']
vis_dir = input_dirs['vis_dir']
filter_out_thresh = input_dirs['filter_out_thresh']

folders_list = input_dirs['folders_list'] # ['LD_LiTian_20201201_9_3'] #, 'LD_FangXM_20201119_1_7', 'LD_FangXM_20201120_3', 'LD_FangXM_20201119_3_7', 'LD_LiTian_20201201_8_7', 'LD_FangXM_20201120_1', 'LD_LiTian_20201201_8_4', 'LD_FangXM_20201119_3_6', 'LD_FangXM_20201120_2']

for folder_name in folders_list: # os.listdir(src_dir)[-1::-1]:
    print(os.path.join(src_dir, folder_name))
    tracking_hairs_id2time_and_mask_dict = {}
    for img_name in sorted(os.listdir(os.path.join(src_dir, folder_name)), key=lambda ele:int(ele.split('.')[0])): # [206:]: # [99:109]:
        print(os.path.join(src_dir, folder_name, img_name))
        single_img_start_time = time.time()
        if not os.path.exists(os.path.join(dst_dir, folder_name)):
            os.mkdir(os.path.join(dst_dir, folder_name))
        curr_img = cv2.imread(os.path.join(src_dir, folder_name, img_name))
        src_full_size_img = copy.deepcopy(curr_img)

        curr_img = cv2.resize(curr_img, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_NEAREST)

        curr_img_full_size = copy.deepcopy(curr_img)
        curr_mask = torch.zeros((int(curr_img.shape[0] / 1), int(curr_img.shape[1] / 1))).byte().to(device) # np.zeros((int(curr_img.shape[0] / 1), int(curr_img.shape[1] / 1))).astype('uint8')
        prev_patch_hair_mask_list = []
        curr_patch_hair_mask_list = []
        # preprocessing
        # curr_img = cv2.resize(curr_img, (int(curr_img.shape[1] / 1), int(curr_img.shape[0] / 3)), interpolation=cv2.INTER_CUBIC)

        # curr_img[:, :, 0] = curr_img_full_size[0:curr_img_full_size.shape[0]:3, :, 0]
        # curr_img[:, :, 1] = curr_img_full_size[1:curr_img_full_size.shape[0]:3, :, 0]
        # curr_img[:, :, 2] = curr_img_full_size[2:curr_img_full_size.shape[0]:3, :, 0]

        # preprocessing end
        curr_img_ori = copy.deepcopy(curr_img)
        resize_start_time1 = time.time()
        curr_img_512 = cv2.resize(curr_img_full_size, (int(512 / curr_img_full_size.shape[0] * curr_img_full_size.shape[1]), 512), interpolation=cv2.INTER_CUBIC)
        resize_end_time1 = time.time()

        mp.set_start_method("spawn", force=True)
        args = get_parser().parse_args()
        setup_logger(name="fvcore")
        logger = setup_logger()
        args.input = [os.path.join(dst_dir, folder_name, img_name)]
        # logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)

        # if len(args.input) == 1:
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"
        img = curr_img_512
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
        del visualized_output
        torch.cuda.empty_cache()
        root_end_time = time.time()

        pred_masks, scores, pred_classes = predictions['instances']._fields['pred_masks'].byte(), predictions['instances']._fields['scores'], predictions['instances']._fields['pred_classes'] # predictions['instances']._fields['pred_masks'].numpy().astype('uint8'), predictions['instances']._fields['scores'].numpy(), predictions['instances']._fields['pred_classes'].numpy()

        del predictions
        torch.cuda.empty_cache()
        # step 2: cut img into patches, during training, step by 1 / 3 bbox, during test, step by 2 / 3 bbox and set a thresholdss for merging two objects from different bboxes with similar slop and position
        assert(len([x for x in pred_classes if x==0])==1)#assert(pred_masks.shape[0] == 1)
        root_mask = pred_masks[[x for x in range(len(pred_classes)) if pred_classes[x]==0][0], :, :]#pred_masks[0, :, :]
        resize_start_time2 = time.time()
        # root_mask = cv2.resize(root_mask, (curr_img_ori.shape[1], curr_img_ori.shape[0]), interpolation=cv2.INTER_NEAREST)
        root_mask = F.upsample_nearest(root_mask.unsqueeze(0).unsqueeze(0).byte(),size=(curr_img_ori.shape[0], curr_img_ori.shape[1])).squeeze(0).squeeze(0)
        resize_end_time2 = time.time()

        del pred_masks, scores, pred_classes
        torch.cuda.empty_cache()

        # hair_mask_list = []
        # for shape in [x for x in curr_json['shapes'] if x['label'] == 'hair']:
        #     segmentation = []
        #     left, right, top, bottom = curr_img_ori.shape[1], 0, curr_img_ori.shape[0], 0
        #     points = shape['points']
        #     for point in points:
        #         segmentation += point
        #         hori, vert = point[0], point[1]
        #         left, right, top, bottom = min([left, hori]), max([right, hori]), min([top, vert]), max([bottom, vert])
        #     hair_mask = labelme.utils.shape_to_mask(curr_img_ori.shape[:2], points)
        #     hair_mask_list.append(hair_mask)
        # hair_mask_list = np.array(hair_mask_list)
        # hair_mask_list = hair_mask_list.astype('uint8')
        # hair_mask_list = hair_mask_list.swapaxes(0, 2).swapaxes(0, 1)

        patch_top_coords = [x for x in range(0, curr_img_ori.shape[0] - img_patch_size, inter_patch_step) if (
                            (torch.sum(root_mask[x:x + img_patch_size, :]) > 0) and (compute_hori_center_gpu(root_mask[x:x + img_patch_size, :]) is not None))]
        # do not miss the bottom block
        added_patch_top_coords_ele = curr_img_ori.shape[0] - img_patch_size
        if (torch.sum(root_mask[added_patch_top_coords_ele:added_patch_top_coords_ele + img_patch_size, :]) > 0) and (compute_hori_center_gpu(root_mask[added_patch_top_coords_ele:added_patch_top_coords_ele + img_patch_size, :]) is not None):
            patch_top_coords.append(added_patch_top_coords_ele)

        patch_column1_left_coords = [max([(compute_hori_center_gpu(root_mask[x:x + img_patch_size, :]) - img_patch_size), 0]) for x in
                                     patch_top_coords if (compute_hori_center_gpu(root_mask[x:x + img_patch_size, :]) is not None)]
        patch_column2_left_coords = [min([(compute_hori_center_gpu(root_mask[x:x + img_patch_size, :]) + img_patch_size), curr_img_ori.shape[1] - 1]) - img_patch_size for x in patch_top_coords if
                                     (compute_hori_center_gpu(root_mask[x:x + img_patch_size, :]) is not None)]
        patch_top_coords = patch_top_coords + patch_top_coords
        patch_left_coords = patch_column1_left_coords + patch_column2_left_coords

        assert(len(patch_top_coords) > 0)
        curr_img_result_dict = {}
        # part_of_prev_patch_hair_mask_list_related_with_curr_patch_hair_mask_list = []
        starting_idx_of_useful_ele_in_part_of_prev_patch_hair_mask_list = 0
        for batch_idx in range(int((len(patch_top_coords) - 1) / inference_batch_size) + 1):
            curr_patch_list = []
            for patch_idx in range(batch_idx * inference_batch_size, min([len(patch_top_coords), (batch_idx + 1) * inference_batch_size])):
                curr_patch_top = int(patch_top_coords[patch_idx])
                curr_patch_left = int(patch_left_coords[patch_idx])
                curr_patch = curr_img_ori[curr_patch_top:curr_patch_top + img_patch_size, curr_patch_left:curr_patch_left + img_patch_size, :]
                curr_patch_list.append(curr_patch)

            hair_start_time = time.time()
            curr_patch_predictions, curr_patch_visualized_output = demo.run_on_image(np.concatenate(curr_patch_list, axis=2), args.confidence_threshold)
            del curr_patch_visualized_output, curr_patch_list
            torch.cuda.empty_cache()
            hair_end_time = time.time()

            # process_time = time.time()
            for patch_idx in range(batch_idx * inference_batch_size, min([len(patch_top_coords), (batch_idx + 1) * inference_batch_size])):
                curr_patch_top = int(patch_top_coords[patch_idx])
                curr_patch_left = int(patch_left_coords[patch_idx])
                if curr_patch_predictions[patch_idx - batch_idx * inference_batch_size]['instances']._fields['pred_masks'].cpu().numpy().astype('uint8').shape[0] == 0:
                    continue

                curr_patch_pred_masks, curr_patch_scores, curr_patch_pred_classes = curr_patch_predictions[patch_idx - batch_idx * inference_batch_size]['instances']._fields['pred_masks'].byte(), \
                                                                                    curr_patch_predictions[patch_idx - batch_idx * inference_batch_size]['instances']._fields['scores'], \
                                                                                    curr_patch_predictions[patch_idx - batch_idx * inference_batch_size]['instances']._fields['pred_classes']

                curr_img_result_dict[patch_idx] = {
                    'curr_patch_top': curr_patch_top,
                    'curr_patch_left': curr_patch_left,
                    'curr_patch_pred_masks': curr_patch_pred_masks,
                    'curr_patch_scores': curr_patch_scores,
                    'curr_patch_pred_classes': curr_patch_pred_classes
                }

                # Why we match hairs from different patches? So as to mutually refine masks
                # Point 2: store instances in different channels, merge channels from different patches
                for hair_idx in range(curr_patch_scores.shape[0]):
                    if curr_patch_scores[hair_idx] >= detection_thresh and curr_patch_pred_classes[hair_idx] == hair_cls:
                        curr_hair_mask = curr_patch_pred_masks[hair_idx]
                        curr_mask[(torch.where(curr_hair_mask > 0)[0] + curr_patch_top, torch.where(curr_hair_mask > 0)[1] + curr_patch_left)] = 255
                        del curr_hair_mask
                        torch.cuda.empty_cache()

                del curr_patch_pred_masks, curr_patch_scores, curr_patch_pred_classes
                torch.cuda.empty_cache()

            del curr_patch_predictions
            torch.cuda.empty_cache()

        write_start_time = time.time()
        if dump_middle_result:
            cv2.imwrite(os.path.join(dst_dir, folder_name, img_name), cv2.resize(curr_mask.cpu().numpy(), (0, 0), fx=1.25, fy=1.25, interpolation=cv2.INTER_NEAREST)) # curr_img = cv2.resize(curr_mask.cpu().numpy(), (0, 0), fx=1.25, fy=1.25, interpolation=cv2.INTER_NEAREST)
        write_end_time = time.time()

        # out_file = open(os.path.join(dst_dir, folder_name, img_name).replace('.jpg', '.json'), "w")
        # json.dump(curr_img_result_dict, out_file)
        # out_file.close()

        # prepare for next stage
        masks_for_hairs = cv2.resize(curr_mask.cpu().numpy(), (0, 0), fx=1.25, fy=1.25, interpolation=cv2.INTER_NEAREST)
        mask_for_root = F.upsample_nearest(root_mask.unsqueeze(0).unsqueeze(0).byte(),size=(src_full_size_img.shape[0], src_full_size_img.shape[1])).squeeze(0).squeeze(0) # cv2.resize(root_mask, (0, 0), fx=1.25, fy=1.25, interpolation=cv2.INTER_NEAREST)

        del root_mask, curr_mask, curr_img_result_dict, prev_patch_hair_mask_list #, curr_patch_hair_mask_list
        torch.cuda.empty_cache()
        single_img_stage1_end_time = time.time()

        #######################################################################################################################################################
        ################################################################ separate the instances ###############################################################
        #######################################################################################################################################################
        # num_hair: number of hairs
        # labels: 1-channel mask, values ranging from 0 to num_hair-1
        # bbox_sizes: num_hair elements, each with (left, top, width, height, area)
        # centers: num_hair elements, center coords
        mask_for_root_cpu = mask_for_root.cpu().numpy()
        kernel = np.ones((25, 25), np.uint8)
        mask_for_root_cpu_eroded = cv2.erode(mask_for_root_cpu, kernel)
        masks_for_hairs[np.where(mask_for_root_cpu_eroded == 1)] = 0 # hair cannot be overlapped with root
        num_hair, labels, bbox_sizes, centers = cv2.connectedComponentsWithStats(masks_for_hairs, connectivity=8)
        # num_hair, bbox_sizes, centers = num_hair - 1, bbox_sizes[1:, :], centers[1:, :]
        # curr_img_patches_json = json.load(open(os.path.join(patch_mask_pred_dir, folder_name, img_name[:-4] + '.json'), 'r'))
        if not os.path.exists(os.path.join(separate_dir, folder_name)):
            os.mkdir(os.path.join(separate_dir, folder_name))
        if not os.path.exists(os.path.join(separate_dir, folder_name, img_name)):
            os.mkdir(os.path.join(separate_dir, folder_name, img_name))
        # prepare for next stage
        track_input = []
        hair_cnt = 0
        labels_gpu = torch.from_numpy(labels).to(device)
        separate_inst_middle_time = time.time()
        for contour_idx in range(1, num_hair):
            curr_contour_left, curr_contour_top, curr_contour_right, curr_contour_bottom = bbox_sizes[contour_idx, 0], bbox_sizes[contour_idx, 1], bbox_sizes[contour_idx, 0]+bbox_sizes[contour_idx, 2], bbox_sizes[contour_idx, 1]+bbox_sizes[contour_idx, 3]
            strip_including_root = mask_for_root[curr_contour_top: curr_contour_bottom]
            # to the left of root
            if len(torch.where(strip_including_root > 0)[1]) == 0:
                continue
            if len(torch.where(strip_including_root > 0)[1]) > 0 and (torch.min(torch.where(strip_including_root > 0)[1]) > curr_contour_right) and (abs(torch.min(torch.where(strip_including_root > 0)[1]) - curr_contour_right) > FA_filter_thresh * abs(curr_contour_right - curr_contour_left)):
                continue
            if len(torch.where(strip_including_root > 0)[1]) > 0 and (torch.max(torch.where(strip_including_root > 0)[1]) < curr_contour_left) and (abs(torch.max(torch.where(strip_including_root > 0)[1]) - curr_contour_left) > FA_filter_thresh * abs(curr_contour_right - curr_contour_left)):
                continue
            if bbox_sizes[contour_idx, 2] < filter_out_thresh and bbox_sizes[contour_idx, 3] < filter_out_thresh:
                continue
            # if an hair is overlapped with root, then it is false alarm
            if torch.sum(mask_for_root[curr_contour_top:curr_contour_bottom, curr_contour_left:curr_contour_right]) > filter_FA_overlap_with_root_thresh * (curr_contour_bottom - curr_contour_top) * (curr_contour_right - curr_contour_left):
                continue

            curr_img_hair_list = torch.zeros(labels.shape).byte().to(device) # np.zeros(labels.shape).astype('uint8')
            # curr_img_hair_list[curr_contour_top:curr_contour_bottom, curr_contour_left:curr_contour_right] = torch.from_numpy(labels[curr_contour_top:curr_contour_bottom, curr_contour_left:curr_contour_right]).to(device)
            curr_img_hair_list[curr_contour_top:curr_contour_bottom, curr_contour_left:curr_contour_right][torch.where(labels_gpu[curr_contour_top:curr_contour_bottom, curr_contour_left:curr_contour_right] == contour_idx)] = 255
            # curr_img_hair_list[torch.where(curr_img_hair_list > 0)] = 255
            # if dump_middle_result:
            #     cv2.imwrite(os.path.join(separate_dir, folder_name, img_name, img_name[:-4] + '_' + str(hair_cnt) + '.png'), curr_img_hair_list.cpu().numpy())
            track_input.append(curr_img_hair_list)
            hair_cnt += 1

        del mask_for_root, masks_for_hairs, curr_img_hair_list, labels_gpu
        torch.cuda.empty_cache()
        single_img_stage2_end_time = time.time()
        #######################################################################################################################################################
        ################################################################ track ################################################################################
        #######################################################################################################################################################
        if sorted(os.listdir(os.path.join(src_dir, folder_name)), key=lambda ele:int(ele.split('.')[0])).index(img_name) == 0 or len(tracking_hairs_id2time_and_mask_dict) == 0:
            for instance_idx in range(len(track_input)):
                curr_instance_mask = track_input[instance_idx]
                vert_mask = torch.where(curr_instance_mask > 0)[0]
                hori_mask = torch.where(curr_instance_mask > 0)[1]
                tracking_hairs_id2time_and_mask_dict[instance_idx] = [sorted(os.listdir(os.path.join(src_dir, folder_name)), key=lambda ele:int(ele.split('.')[0])).index(img_name), \
                                                                      vert_mask, hori_mask]
        else:
            # compute matching matrix
            matching_matrix = np.zeros((len(tracking_hairs_id2time_and_mask_dict), len(track_input)))
            curr_img_masks_list = []
            else_start = time.time()

            for instance_idx in range(len(track_input)):
                debug_start = time.time()
                curr_instance_mask = track_input[instance_idx]
                vert_mask = torch.where(curr_instance_mask > 0)[0]
                hori_mask = torch.where(curr_instance_mask > 0)[1]
                curr_img_masks_list.append([vert_mask, hori_mask])
                debug_middle = time.time()
                for tracking_hairs_id2time_and_mask_dict_idx in range(len(tracking_hairs_id2time_and_mask_dict)):
                    matching_matrix[tracking_hairs_id2time_and_mask_dict_idx, instance_idx] = \
                        ot_iou(tracking_hairs_id2time_and_mask_dict[[x for x in tracking_hairs_id2time_and_mask_dict][tracking_hairs_id2time_and_mask_dict_idx]], [vert_mask, hori_mask], curr_instance_mask.shape[0], curr_instance_mask.shape[1])
                debug_end = time.time()
            ot_time_start = time.time()
            prev_curr_transportation_array = ot_matching_matrix(1.0 - matching_matrix)
            ot_time_end = time.time()
            # print('ot time: ' + str(ot_time_end - ot_time_start))
            # matching_matrix is a floating matrix, prev_curr_transportation_array is a 0/1 matrix
            matching_matrix = np.multiply(matching_matrix, prev_curr_transportation_array)
            # update bases
            update_time_start = time.time()
            new_hair_list = range(matching_matrix.shape[1])
            for prev_hair_idx in range(matching_matrix.shape[0]):
                if np.max(matching_matrix[prev_hair_idx, :]) >= update_tracking_hair_thresh:
                    tracking_hairs_id2time_and_mask_dict[[x for x in tracking_hairs_id2time_and_mask_dict][prev_hair_idx]][0] = sorted(os.listdir(os.path.join(src_dir, folder_name)), key=lambda ele:int(ele.split('.')[0])).index(img_name)
                    tracking_hairs_id2time_and_mask_dict[[x for x in tracking_hairs_id2time_and_mask_dict][prev_hair_idx]][1] = curr_img_masks_list[np.argmax(matching_matrix[prev_hair_idx, :]).tolist()][0]
                    tracking_hairs_id2time_and_mask_dict[[x for x in tracking_hairs_id2time_and_mask_dict][prev_hair_idx]][2] = curr_img_masks_list[np.argmax(matching_matrix[prev_hair_idx, :]).tolist()][1]
                    new_hair_list = [x for x in new_hair_list if x != np.argmax(matching_matrix[prev_hair_idx, :])]
            for new_hair_list_idx in range(len(new_hair_list)):
                tracking_hairs_id2time_and_mask_dict[max([x for x in tracking_hairs_id2time_and_mask_dict]) + new_hair_list_idx + 1] = [sorted(os.listdir(os.path.join(src_dir, folder_name)), key=lambda ele:int(ele.split('.')[0])).index(img_name),
                                                                                                                                           curr_img_masks_list[new_hair_list[new_hair_list_idx]][0],
                                                                                                                                           curr_img_masks_list[new_hair_list[new_hair_list_idx]][1]]
            update_time_end = time.time()
        ##################################################### tracking end ##########################################################################
        ############################ draw ###########################################################################################################
        curr_img_instance_id_list = {}
        for instance_idx in tracking_hairs_id2time_and_mask_dict:
            curr_instance_mask = np.zeros((src_full_size_img.shape[0], src_full_size_img.shape[1])) # .byte().to(device)
            vert_coord = tracking_hairs_id2time_and_mask_dict[instance_idx][1]
            hori_coord = tracking_hairs_id2time_and_mask_dict[instance_idx][2]
            curr_instance_mask[vert_coord.cpu().numpy(), hori_coord.cpu().numpy()] = 255

            curr_inst_top, curr_inst_left, curr_inst_bottom, curr_inst_right = max([torch.min(vert_coord).item() - 5, 0]), \
                                       max([torch.min(hori_coord).item() - 5, 0]), \
                                       min([torch.max(vert_coord).item() + 5, curr_instance_mask.shape[0] - 1]), \
                                       min([torch.max(hori_coord).item() + 5, curr_instance_mask.shape[1] - 1])

            kernel = np.ones((5, 5), np.uint8)
            boundary = np.array(cv2.morphologyEx(curr_instance_mask[curr_inst_top:curr_inst_bottom, curr_inst_left:curr_inst_right], cv2.MORPH_GRADIENT, kernel))
            src_full_size_img[:, :, 2][np.where(boundary > 0)[0] + curr_inst_top, np.where(boundary > 0)[1] + curr_inst_left] = 255
            src_full_size_img[:, :, 0][np.where(boundary > 0)[0] + curr_inst_top, np.where(boundary > 0)[1] + curr_inst_left] = 0
            src_full_size_img[:, :, 1][np.where(boundary > 0)[0] + curr_inst_top, np.where(boundary > 0)[1] + curr_inst_left] = 0

            curr_img_instance_id_list[instance_idx] = [torch.min(vert_coord).item(), torch.min(hori_coord).item()] # [np.array(np.min(vert_coord)).tolist(), np.array(np.min(hori_coord)).tolist()]
        curr_img_write = np.array(src_full_size_img)
        for curr_img_instance_id_list_key in curr_img_instance_id_list:
            curr_img_write = cv2.putText(curr_img_write, str(curr_img_instance_id_list_key), (curr_img_instance_id_list[curr_img_instance_id_list_key][1], curr_img_instance_id_list[curr_img_instance_id_list_key][0]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        if not os.path.exists(os.path.join(vis_dir, folder_name)):
            os.mkdir(os.path.join(vis_dir, folder_name))
        cv2.imwrite(os.path.join(vis_dir, folder_name, img_name), curr_img_write)

        del curr_img_instance_id_list, track_input
        torch.cuda.empty_cache()

        print('success')
        single_img_end_time = time.time()
        print(str(single_img_stage1_end_time - single_img_start_time) + ' ' + str(single_img_stage2_end_time-single_img_stage1_end_time) + ' ' + str(single_img_end_time - single_img_stage2_end_time))

