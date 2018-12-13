import numpy as np
from tf_utils import visualization_utils_cv2 as vis_util
from tf_utils import ops as utils_ops
from lib.session_worker import SessionWorker
from lib.mtload_graph_mask_v1 import LoadFrozenGraph
from lib.load_label_map import LoadLabelMap
from lib.mpvariable import MPVariable
from lib.mpvisualizeworker import MPVisualizeWorker, visualization
from lib.mpio import start_sender
from lib.color_map import STANDARD_COLORS_ARRAY

import time
import cv2
import tensorflow as tf
import os

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue


class MASKV1():
    def __init__(self):
        return

    def start(self, cfg):
        """ """ """ """ """ """ """ """ """ """ """
        GET CONFIG
        """ """ """ """ """ """ """ """ """ """ """
        FORCE_GPU_COMPATIBLE = cfg['force_gpu_compatible']
        SAVE_TO_FILE         = cfg['save_to_file']
        VISUALIZE            = cfg['visualize']
        VIS_WORKER           = cfg['vis_worker']
        VIS_TEXT             = cfg['vis_text']
        MAX_FRAMES           = cfg['max_frames']
        WIDTH                = cfg['width']
        HEIGHT               = cfg['height']
        FPS_INTERVAL         = cfg['fps_interval']
        DET_INTERVAL         = cfg['det_interval']
        DET_TH               = cfg['det_th']
        WORKER_THREADS       = cfg['worker_threads']
        LOG_DEVICE           = cfg['log_device']
        ALLOW_MEMORY_GROWTH  = cfg['allow_memory_growth']
        DEBUG_MODE           = cfg['debug_mode']
        LABEL_PATH           = cfg['label_path']
        NUM_CLASSES          = cfg['num_classes']
        SRC_FROM             = cfg['src_from']
        CAMERA = 0
        MOVIE  = 1
        IMAGE  = 2
        if SRC_FROM == 'camera':
            SRC_FROM = CAMERA
            VIDEO_INPUT = cfg['camera_input']
        elif SRC_FROM == 'movie':
            SRC_FROM = MOVIE
            VIDEO_INPUT = cfg['movie_input']
        elif SRC_FROM == 'image':
            SRC_FROM = IMAGE
            VIDEO_INPUT = cfg['image_input']
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        LOAD FROZEN_GRAPH
        """ """ """ """ """ """ """ """ """ """ """
        load_frozen_graph = LoadFrozenGraph(cfg)
        graph = load_frozen_graph.load_graph()
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        LOAD LABEL MAP
        """ """ """ """ """ """ """ """ """ """ """
        llm = LoadLabelMap()
        category_index = llm.load_label_map(cfg)
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        PREPARE TF CONFIG OPTION
        """ """ """ """ """ """ """ """ """ """ """
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=LOG_DEVICE)
        config.gpu_options.allow_growth = ALLOW_MEMORY_GROWTH
        config.gpu_options.force_gpu_compatible = FORCE_GPU_COMPATIBLE
        #config.gpu_options.per_process_gpu_memory_fraction = 0.01 # 80MB memory is enough to run on TX2
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        PREPARE GRAPH I/O TO VARIABLE
        """ """ """ """ """ """ """ """ """ """ """
        # Define Input and Ouput tensors
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')
        detection_masks = graph.get_tensor_by_name('detection_masks:0')


        """ """ """ """ """ """ """ """ """ """ """
        START CAMERA
        """ """ """ """ """ """ """ """ """ """ """
        if SRC_FROM == CAMERA:
            from lib.webcam import WebcamVideoStream as VideoReader
        elif SRC_FROM == MOVIE:
            from lib.video import VideoReader
        elif SRC_FROM == IMAGE:
            from lib.image import ImageReader as VideoReader
        video_reader = VideoReader()

        if SRC_FROM == IMAGE:
            video_reader.start(VIDEO_INPUT, save_to_file=SAVE_TO_FILE)
            frame_cols, frame_rows = HEIGHT, WIDTH
        else: # CAMERA, MOVIE
            video_reader.start(VIDEO_INPUT, WIDTH, HEIGHT, save_to_file=SAVE_TO_FILE)
            frame_cols, frame_rows = video_reader.getSize()
            """ STATISTICS FONT """
            fontScale = frame_rows/1000.0
            if fontScale < 0.4:
                fontScale = 0.4
            fontThickness = 1 + int(fontScale)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        if SRC_FROM == MOVIE:
            dir_path, filename = os.path.split(VIDEO_INPUT)
            filepath_prefix = filename
        elif SRC_FROM == CAMERA:
            filepath_prefix = 'frame'
        """ """


        """ """ """ """ """ """ """ """ """ """ """
        PREAPRE GRAPH MASK OUTPUT
        """ """ """ """ """ """ """ """ """ """ """
        # The following processing is only for single image
        _detection_boxes = tf.squeeze(detection_boxes, [0])
        _detection_masks = tf.squeeze(detection_masks, [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        _real_num_detection = tf.cast(num_detections[0], tf.int32)
        _detection_boxes = tf.slice(_detection_boxes, [0, 0], [_real_num_detection, -1])
        _detection_masks = tf.slice(_detection_masks, [0, 0, 0], [_real_num_detection, -1, -1])
        _detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            _detection_masks, _detection_boxes, frame_rows, frame_cols)
        _detection_masks_reframed = tf.cast(
            tf.greater(_detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        detection_masks = tf.expand_dims(
            _detection_masks_reframed, 0)
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        START WORKER THREAD
        """ """ """ """ """ """ """ """ """ """ """
        workers = []
        worker_tag = 'worker'
        # create session worker threads
        for i in range(WORKER_THREADS):
            workers += [SessionWorker(worker_tag, graph, config)]
        worker_opts = [detection_boxes, detection_scores, detection_classes, num_detections, detection_masks]
        """ """

        """
        START VISUALIZE WORKER
        """
        if VISUALIZE and VIS_WORKER:
            q_out = Queue.Queue()
            vis_worker = MPVisualizeWorker(cfg, MPVariable.vis_in_con)
            """ """ """ """ """ """ """ """ """ """ """
            START SENDER THREAD
            """ """ """ """ """ """ """ """ """ """ """
            start_sender(MPVariable.det_out_con, q_out)
        proc_frame_counter = 0
        vis_proc_time = 0


        """ """ """ """ """ """ """ """ """ """ """
        WAIT UNTIL THE FIRST DUMMY IMAGE DONE
        """ """ """ """ """ """ """ """ """ """ """
        print('Loading...')
        sleep_interval = 0.1

        """ """ """ """ """ """ """ """ """ """ """
        DETECTION LOOP
        """ """ """ """ """ """ """ """ """ """ """
        print('Starting Detection')
        sleep_interval = 0.005
        top_in_time = None
        frame_in_processing_counter = 0
        current_in_worker_id = -1
        worker_id_queue = Queue.Queue()
        retry_worker_id_queue = Queue.Queue()
        try:
            if not video_reader.running:
                raise IOError(("Input src error."))
            while MPVariable.running.value:
                if top_in_time is None:
                    top_in_time = time.time()
                """
                RUN ALL WORKERS
                """
                if video_reader.running:
                    for i in range(WORKER_THREADS):
                        worker_in_id = i + current_in_worker_id + 1
                        worker_in_id %= WORKER_THREADS
                        if workers[worker_in_id].is_sess_empty(): # must need for speed
                            cap_in_time = time.time()
                            if SRC_FROM == IMAGE:
                                frame, filepath = video_reader.read()
                                if frame is not None:
                                    frame_in_processing_counter += 1
                                    frame = cv2.resize(frame, (frame_cols, frame_rows))
                            else:
                                frame = video_reader.read()
                                if frame is not None:
                                    filepath = filepath_prefix+'_'+str(proc_frame_counter)+'.png'
                                    frame_in_processing_counter += 1
                            if frame is not None:
                                image_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0) # np.expand_dims is faster than []
                                #image_expanded = np.expand_dims(frame, axis=0) # BGR image for input. Of couse, bad accuracy in RGB trained model, but speed up.
                                cap_out_time = time.time()
                                # put new queue
                                worker_feeds = {image_tensor: image_expanded}
                                worker_extras = {'image':frame, 'top_in_time':top_in_time, 'cap_in_time':cap_in_time, 'cap_out_time':cap_out_time, 'filepath': filepath} # always image draw.
                                workers[worker_in_id].put_sess_queue(worker_opts, worker_feeds, worker_extras)
                                current_in_worker_id = worker_in_id
                                worker_id_queue.put(worker_in_id)
                                time.sleep(sleep_interval*10/WORKER_THREADS)
                            break
                elif frame_in_processing_counter <= 0:
                    MPVariable.running.value = False
                    break

                q = None
                if not retry_worker_id_queue.empty():
                    #print("retry!")
                    worker_out_id = retry_worker_id_queue.get(block=False)
                    worker_out_id %= WORKER_THREADS
                    retry_worker_id_queue.task_done()
                    q = workers[worker_out_id].get_result_queue()
                    if q is None:
                        retry_worker_id_queue.put(worker_out_id)
                elif not worker_id_queue.empty():
                    worker_out_id = worker_id_queue.get(block=False)
                    worker_out_id %= WORKER_THREADS
                    worker_id_queue.task_done()
                    q = workers[worker_out_id].get_result_queue()
                    if q is None:
                        retry_worker_id_queue.put(worker_out_id)
                if q is None:
                    # detection is not complete yet. ok nothing to do.
                    time.sleep(sleep_interval)
                    continue
                #print("ok!")

                frame_in_processing_counter -= 1
                boxes, scores, classes, num, masks, extras = q['results'][0], q['results'][1], q['results'][2], q['results'][3], q['results'][4], q['extras']
                boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
                det_out_time = time.time()

                """
                ALWAYS BOX DRAW ON IMAGE
                """
                vis_in_time = time.time()
                image = extras['image']
                if SRC_FROM == IMAGE:
                    filepath = extras['filepath']
                    frame_rows, frame_cols = image.shape[:2]
                    """ STATISTICS FONT """
                    fontScale = frame_rows/1000.0
                    if fontScale < 0.4:
                        fontScale = 0.4
                    fontThickness = 1 + int(fontScale)
                else:
                    filepath = extras['filepath']
                image = visualization(category_index, image, boxes, scores, classes, DEBUG_MODE, VIS_TEXT, FPS_INTERVAL,
                                      fontFace=fontFace, fontScale=fontScale, fontThickness=fontThickness, masks=masks)

                """
                VISUALIZATION
                """
                if VISUALIZE:
                    if (MPVariable.vis_skip_rate.value == 0) or (proc_frame_counter % MPVariable.vis_skip_rate.value < 1):
                        if VIS_WORKER:
                            q_out.put({'image':image, 'vis_in_time':vis_in_time})
                        else:
                            #np.set_printoptions(precision=5, suppress=True, threshold=np.inf)  # suppress scientific float notation
                            """
                            SHOW
                            """
                            cv2.imshow("Object Detection", image)
                            # Press q to quit
                            if cv2.waitKey(1) & 0xFF == 113: #ord('q'):
                                break
                            MPVariable.vis_frame_counter.value += 1
                            vis_out_time = time.time()
                            """
                            PROCESSING TIME
                            """
                            vis_proc_time = vis_out_time - vis_in_time
                            MPVariable.vis_proc_time.value += vis_proc_time
                else:
                    """
                    NO VISUALIZE
                    """
                    for box, score, _class in zip(boxes, scores, classes):
                        if proc_frame_counter % DET_INTERVAL == 0 and score > DET_TH:
                            label = category_index[_class]['name']
                            print("label: {}\nscore: {}\nbox: {}".format(label, score, box))

                    vis_out_time = time.time()
                    """
                    PROCESSING TIME
                    """
                    vis_proc_time = vis_out_time - vis_in_time

                if SAVE_TO_FILE:
                    if SRC_FROM == IMAGE:
                        video_reader.save(image, filepath)
                    else:
                        video_reader.save(image)

                proc_frame_counter += 1
                if proc_frame_counter > 100000:
                    proc_frame_counter = 0
                """
                PROCESSING TIME
                """
                top_in_time = extras['top_in_time']
                cap_proc_time = extras['cap_out_time'] - extras['cap_in_time']
                worker_proc_time = extras['worker_out_time'] - extras['worker_in_time']
                lost_proc_time = det_out_time - top_in_time - cap_proc_time - worker_proc_time
                total_proc_time = det_out_time - top_in_time
                MPVariable.cap_proc_time.value += cap_proc_time
                MPVariable.worker_proc_time.value += worker_proc_time
                MPVariable.lost_proc_time.value += lost_proc_time
                MPVariable.total_proc_time.value += total_proc_time

                if DEBUG_MODE:
                    sys.stdout.write('snapshot FPS:{: ^5.1f} total:{: ^10.5f} cap:{: ^10.5f} worker:{: ^10.5f} lost:{: ^10.5f} | vis:{: ^10.5f}\n'.format(
                        MPVariable.fps.value, total_proc_time, cap_proc_time, worker_proc_time, lost_proc_time, vis_proc_time))
                """
                EXIT WITHOUT GUI
                """
                if not VISUALIZE and MAX_FRAMES > 0:
                    if proc_frame_counter >= MAX_FRAMES:
                        MPVariable.running.value = False
                        break

                """
                CHANGE SLEEP INTERVAL
                """
                if MPVariable.frame_counter.value == 0 and MPVariable.fps.value > 0:
                    sleep_interval = 0.1 / MPVariable.fps.value
                    MPVariable.sleep_interval.value = sleep_interval
                MPVariable.frame_counter.value += 1
                top_in_time = None
            """
            END while
            """
        except KeyboardInterrupt:
            pass
        except:
            import traceback
            traceback.print_exc()
        finally:
            """ """ """ """ """ """ """ """ """ """ """
            CLOSE
            """ """ """ """ """ """ """ """ """ """ """
            if VISUALIZE and VIS_WORKER:
                q_out.put(None)
            MPVariable.running.value = False
            for i in range(WORKER_THREADS):
                workers[i].stop()
            video_reader.stop()

            if VISUALIZE:
                cv2.destroyAllWindows()
            """ """

        return

