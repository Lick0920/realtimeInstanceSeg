# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
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


# constants
WINDOW_NAME = "COCO detections"


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
        # default="D:\\project_python\\detectron2\\configs\\quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        # default = 'D:\\project_python\\SparseInst\\configs\\a3d2048_down.yaml',
        # default = 'D:\\project_python\\SparseInst\\configs\\3d_test\\3d_dep_2k3k_patchall.yaml',
        # default = 'D:\\project_python\\SparseInst\\configs\\3d_test\\3d_dep_2k3k_qiekuai.yaml',
        # default= 'D:\\project_python\\SparseInst\\configs\\3d_test\\3d_2k3k_HRnet.yaml',
        # default= 'D:\\project_python\\SparseInst\\configs\\arootair_resr50vd_dcn_giam_aug.yaml',
        # default = 'D:\\project_python\\SparseInst\\output_retry\\3d_2k3k_4x4\\config.yaml',
        # default='D:\\project_python\\SparseInst\\output_retry\\a2kinput_ori_downchannel_retry_re4999_preloss2.0_lre-1\\config.yaml',
        # default='D:\\project_python\\SparseInst\\output_retry\\houjiangcaiyang_23TT\\config.yaml',
        # default='D:\\project_python\\SparseInst\\output_retry\\resnet3d_notdep\\config.yaml',
        default='D:\\project_python\\SparseInst\\output_cityscapes\\2d_base_res2345\\config.yaml',
        # default='D:\\project_python\\SparseInst\\output_retry\\resnet2d_SPD_2048\\config.yaml',
        # default='D:\\project_python\\SparseInst\\output_retry\\resnet2d_baseline_notpre\\config.yaml',
        # default='D:\\project_python\\SparseInst\\output\\aaaresnet3d_bs8_cocopretrain_jiangcaiyang\\config.yaml',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        # default=["E:\\dataset\\coco\\val2017\\000000000139.jpg"],
        # default= ['E:\\dataset\\roothair\\train_ori\\000000000008.jpg'],
        default= ["E:\\dataset\\cityscapes\\leftImg8bit\\val\\frankfurt\\frankfurt_000000_000294_leftImg8bit.png"],
        # default= ['E:\dataset\\roothair\\train_2k3k\\000000000004_3990_918.jpg'],
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
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output\\a2kinput_ori_downchannel_resume0.9\\model_0001999.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output_retry\\a2kinput_ori_downchannel_retry_re4999_preloss2.0_lre-1\\model_0021999.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output_retry\\a2kinput_ori_down_qiekuai\\model_0001199.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output_retry\\3d_2k3k_hrnet\\model_0023199.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output\\aaaresnet3d_bs8_cocopretrain_jiangcaiyang\\model_0004999.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output_retry\\3d_2k3k_4x4_lr100\\model_0000399.pth'],
        # default = ['MODEL.WEIGHTS', 'D:\\project_python\\SparseInst\\output_retry\\a2kinput_ori_downchannel_retry_re4999_preloss2.0_lre-1\\model_0019999.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output_retry\\houjiangcaiyang_23TT_lrwarmup\\model_0037399.pth'],
        # default=["MODEL.WEIGHTS","D:\\project_python\\SparseInst\\output_3d_rootair\\base_2FTT\\model_0002399.pth"],
        default=["MODEL.WEIGHTS","D:\\project_python\\SparseInst\\output_cityscapes\\2d_base_res2345\\model_0011499.pth"],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output_retry\\resnet2d_SPD_2048\\model_0001199.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\output_retry\\resnet2d_baseline_notpre\\model_0000599.pth'],
        # default=["MODEL.WEIGHTS", 'D:\\project_python\\SparseInst\\model_0079999.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, parallel=False)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            #             img = read_image(path, format="BGR")
            # OneNet uses RGB input as default
            img = read_image(path, format="RGB")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(
                img, args.confidence_threshold)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(
                        len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(
                        args.output, os.path.basename(path))
                else:
                    assert len(
                        args.output) > 0, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(
                    WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam, args.confidence_threshold)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video, args.confidence_threshold), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
