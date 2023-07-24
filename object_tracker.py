import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, './yolov5')
# general utils
import os
# fixes error: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-when-fitting-models
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
# Deep nets
import cv2
import torch
import torch.backends.cudnn as cudnn
from yolov5.models.experimental import attempt_load
# from yolov5.utils.google_utils import attempt_download # before update
from yolov5.utils.downloads import attempt_download # after update
# from yolov5.utils.datasets import LoadImages, LoadStreams # before update
from yolov5.utils.dataloaders import LoadImages, LoadStreams # after update
# from yolov5.utils.torch_utils import select_device, time_synchronized # before update
from yolov5.utils.torch_utils import select_device, time_sync # after update
# from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow # before update
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes, check_imshow # after update
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
# Object Character Recognition (OCR)
import easyocr

# csv.field_size_limit(sys.maxsize)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def xyxy_to_xywh(*xyxy):
    """"Calculates the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def detect(**kwargs):
    '''
    :param yolo_weights: model.pt path
    :param deep_sort_weights: ckpt.t7 path
    :param conf_thres: object confidence threshold
    :param iou_thres: IOU threshold for NMS
    :param fourcc: output video codec (verify ffmpeg support)
    :param device: cuda device, i.e. 0 or 0,1,2,3 or cpu
    :param augment: augmented inference
    :param agnostic_nms: class-agnostic NMS
    :param config_deepsort: 
    :param kwargs: 
    :return: 
    '''

    video_name = kwargs['video_name']
    source = kwargs['source']
    device = kwargs['device']
    yolo_weights = kwargs['yolo_weights']
    deep_sort_weights = kwargs['deep_sort_weights']
    config_deepsort = kwargs['config_deepsort']
    show_vid = kwargs['show_vid']
    save_vid = kwargs['save_vid']
    save_txt = kwargs['save_txt']
    imgsz = kwargs['img_size']
    conf_thres = kwargs['conf_thres']
    iou_thres = kwargs['iou_thres']
    agnostic_nms = kwargs['agnostic_nms']
    augment = kwargs['augment']
    classes = kwargs['classes']
    OCR_PER_SEC = kwargs['ocr_per_sec']
    OUTPUT_VIDEO_PATH = kwargs['output_video_path']
    TRACKLOG_PATH = kwargs['track_log']
    OCR_LOG_PATH = kwargs['ocr_log']
    VIDEOS_METADATA_LOG = kwargs['videos_metadata_log']

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    # Initialize
    device = select_device(device)
    # list to store images that are written to video file
    video_img_storage = []
    ## The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    ## its own .txt file. Hence, in that case, the output folder is not restored
    # if not evaluate:
    #     if os.path.exists(out):
    #         shutil.rmtree(out)  # delete output folder
    #     os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    # model = attempt_load(yolo_weights, map_location=device)  # load FP32 model # before update
    model = attempt_load(yolo_weights, device=device)  # load FP32 model # after update
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names and colors
    if half:
        model.half()  # to FP16

    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow(warn=True)
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    # NEW initialise easyocr + logging
    ocr_reader = easyocr.Reader(['fr', 'en', 'de'], gpu=True)  # need to run only once to load model into memory

    fps = None
    width = None
    height = None
    for frame_idx, (path, img, im0s, vid_cap, vid_cap_str) in enumerate(dataset):
        if frame_idx == 0:
            fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # calculate based on OCR_PER_SEC the frame interval of OCR processing
            OCR_NTH_FRAME = round(fps / OCR_PER_SEC)
            # write video specific metadata to file
            delimiter = ';'
            with open(VIDEOS_METADATA_LOG, 'at', encoding='utf-8') as f:
                f.write(f'{video_name}{delimiter}{fps}{delimiter}{width}{delimiter}{height}{delimiter}{OCR_NTH_FRAME}\n')

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        # t1 = time_synchronized() # before update
        t1 = time_sync() # after update
        pred = model(img, augment=augment)[0]

        # NEW apply OCR to frame and save detected text in OCR_logfile
        # im0s must be taken as input or the plain image path variable 'path' but this is not feasible with videos
        # reason not exactly known, why for instance variable img does not work - probably preprocessing
        # does not work for WEBCAM!
        # --- img_for_ocr = np.array(im0s) -- not necessary

        img_for_ocr = im0s
        ## rescale image to only include the top half of the image (higher probability of street names) to improve performance (tested by saving image before after - functional!)
        # torchvision.transforms.ToPILImage()(img_for_ocr).save('img_before.png')
        # fetching the dimensions
        height, width, depth = img_for_ocr.shape
        # keep only top half of image
        img_for_ocr = img_for_ocr.reshape(height, width, depth)
        img_for_ocr = img_for_ocr[:int(height/2),:,:]
        # torchvision.transforms.ToPILImage()(img_for_ocr).save('img_after.png')
        # check if OCR shall be performed on the current frame

        # if int(frame_idx) % int(OCR_NTH_FRAME) == 0:
        #     ocr_result = ocr_reader.readtext(img_for_ocr, detail=1)
        #     if len(ocr_result) != 0:
        #         with open(OCR_LOG_PATH, 'a', encoding='utf-8') as f:
        #             for result in ocr_result:
        #                 bbox = result[0]
        #                 bbox_bl = bbox[0]
        #                 bbox_tr = bbox[2]
        #                 bbox_xmin = bbox_bl[0]
        #                 bbox_ymin = bbox_bl[1]
        #                 bbox_xmax = bbox_tr[0]
        #                 bbox_ymax = bbox_tr[1]
        #                 text = result[1].replace(';', ' ') # replaces all csv separator's in text!
        #                 certainty = result[2]
        #                 # here potentially set certainty threshold!!
        #                 ocr_entry = '{};{};{};{};{};{};{}\n'.format(frame_idx, text, certainty, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        #                 f.write(ocr_entry)
        # apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        # t2 = time_synchronized() # before update
        t2 = time_sync() # after update
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, print_string, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, print_string, im0 = path, '', im0s

            print_string += '%gx%g ' % img.shape[2:]  # print string
            # save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() # before update
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round() # after update
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    print_string += '%g %ss, ' % (n, names[int(c)])  # add to string
                # just again for getting all class_names (without removing duplicates)
                class_names = []
                for c in det[:, -1]:
                    class_names.append(int(c))

                xywh_bboxs = []
                confs = []
                # adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0, class_names)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    class_names = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
                    # Write MOT compliant results to file
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-2]
                            class_name = int(output[-1])
                            with open(TRACKLOG_PATH, 'a', encoding='utf-8') as f:
                                f.write(f'{frame_idx};{identity};{class_name};{bbox_top};{bbox_left};{bbox_w};{bbox_h}\n')
            else:
                deepsort.increment_ages()
            # print time (inference + NMS)
            # add frame info to print_string
            print_string = vid_cap_str + ' ' + print_string
            print(f'{print_string}Done. {round(t2 - t1, 2)}', end='\r')
            # stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

        # save results (video with detections) to a list and write to video file later
        if save_vid:
            video_img_storage.append(im0)

    if save_vid:
        vid_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame_idx, img_file in enumerate(video_img_storage):
            vid_writer.write(img_file)
            print(f'frame_idx: {frame_idx} ,fps: {fps}, height: {height}, width: {width}', end='\r')

    # if save_txt or save_vid:
    #     print('[+] Results saved to %s' % os.getcwd() + os.sep + out)
    #     if platform == 'darwin':  # MacOS
    #         os.system('open ' + save_path)

    # # NEW: copy existing output to persistent ./output folder
    # for file in os.listdir(out):
    #     shutil.move(os.path.join(out, file), os.path.join('./output/', file))

    print('[*] Done. (%.3fs)' % (time.time() - t0))

