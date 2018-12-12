import multiprocessing
import logging
import time
import sys
import numpy as np
import threading
from lib.mpvariable import MPVariable
import types
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import copy_reg
elif PY3:
    import copyreg as copy_reg

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class FPS():
    def __init__(self, cfg):
        self.cfg = cfg
        return

    def start_counter(self):
        """
        Start via Process
        """
        m = multiprocessing.Process(target=self.process_fps_counter, args=())
        m.start()
        return m

    def start_console(self):
        """
        Start via Process
        """
        m = multiprocessing.Process(target=self.process_fps_console, args=())
        m.start()
        return m

    def process_fps_counter(self):
        """
        frame_counter.value: Frame counter value shared by processes.
        update fps by 0.2 sec
        """
        logging.debug("enter")
        FPS_INTERVAL         = self.cfg['fps_interval']
        DEBUG_MODE           = self.cfg['debug_mode']
        MAX_VIS_FPS          = self.cfg['max_vis_fps']

        sleep_interval = 1.0 / 50.0 # Wakeup and time checks N times per sec.
        snapshot_interval = 1.0 / 5.0 # FPS calculate N times per sec. (must snapshot_interval >= sleep_interval)
        fps_stream_length = FPS_INTERVAL # FPS stream seconds length. FPS is the frames per second processed during the most recent this time.
        fps_stream = [] # Array of fps_snapshot during fps_stream_length
        snapshot = None # One array (frames_in_interval, interval_seconds, unixtime) per snapshot_interval
        vis_fps_stream = []
        vis_snapshot = None

        if DEBUG_MODE:
            """ FOR PERFORMANCE DEBUG """
            snapshot_fps = 0
            min_snapshot_fps = 10000
            max_snapshot_fps = 0
            min_snapshot_list = []
            max_snapshot_list = []
            """ """
        try:
            launch_time = time.time()
            while MPVariable.running.value and MPVariable.frame_counter.value == 0:
                time.sleep(sleep_interval)
            MPVariable.first_complete_time.value = time.time() - launch_time
            print("Time to first image:{}".format(MPVariable.first_complete_time.value))

            previos_work_time = time.time()
            while MPVariable.running.value:
                time.sleep(sleep_interval)
                now_time = time.time()
                if now_time >= previos_work_time + snapshot_interval:
                    ### FPS update by snapshot_interval ###
                    snapshot_frames = MPVariable.frame_counter.value
                    MPVariable.frame_counter.value = 0
                    vis_snapshot_frames = MPVariable.vis_frame_counter.value
                    MPVariable.vis_frame_counter.value = 0

                    """
                    FPS stream
                    """
                    snapshot_seconds = now_time - previos_work_time
                    snapshot = (snapshot_frames, snapshot_seconds, now_time)
                    fps_stream += [snapshot]
                    vis_snapshot = (vis_snapshot_frames, snapshot_seconds, now_time)
                    vis_fps_stream += [vis_snapshot]

                    if DEBUG_MODE:
                        """
                        FOR PERFORMANCE DEBUG
                        FPS snapshot calculation
                        """
                        snapshot_fps = snapshot_frames/snapshot_seconds
                        MPVariable.fps_snapshot.value = snapshot_fps # FPS of snapshot. for visualize
                        if min_snapshot_fps >= snapshot_fps:
                            min_snapshot_fps = snapshot_fps
                            min_snapshot_list += [snapshot_fps]
                            print("min_snapshot:{:.1f} {}".format(snapshot_fps, snapshot))
                        if max_snapshot_fps <= snapshot_fps:
                            max_snapshot_fps = snapshot_fps
                            max_snapshot_list += [snapshot_fps]
                            print("max_snapshot:{:.1f} {}".format(snapshot_fps, snapshot))
                        """ """

                    """
                    PROC FPS
                    """
                    while MPVariable.running.value:
                        (min_frame, min_seconds, min_time) = fps_stream[0]
                        if now_time - min_time > fps_stream_length:
                            """
                            drop old snapshot
                            """
                            fps_stream.pop(0)
                        else:
                            """
                            goto FPS streaming calculation
                            """
                            break
                    if(len(fps_stream) > 0):
                        """
                        FPS streaming calculation
                        count frames and seconds in stream
                        """
                        np_fps_stream = np.array(fps_stream)
                        np_fps_stream = np_fps_stream[:,:2]    # take frames, seconds. drop unixtime.
                        np_fps = np.sum(np_fps_stream, axis=0) # [total_frames, total_seconds] duaring fps_stream_length

                        """
                        insert local values to shared variables
                        """
                        MPVariable.fps_frames.value = int(np_fps[0]) # for console output
                        MPVariable.fps_seconds.value = np_fps[1]     # for console output
                        MPVariable.fps.value = np_fps[0]/np_fps[1]   # for visualize and console
                    else:
                        MPVariable.fps_frames.value = 0
                        MPVariable.fps_seconds.value = -1 # for toooooooo slow fps check. if -1 sec appears, fps_stream_length should set more long time.
                        MPVariable.fps.value = 0

                    """
                    VIS FPS
                    """
                    while MPVariable.running.value:
                        (min_frame, min_seconds, min_time) = vis_fps_stream[0]
                        if now_time - min_time > fps_stream_length:
                            """
                            drop old snapshot
                            """
                            vis_fps_stream.pop(0)
                        else:
                            """
                            goto FPS streaming calculation
                            """
                            break
                    if(len(vis_fps_stream) > 0):
                        """
                        FPS streaming calculation
                        count frames and seconds in stream
                        """
                        np_fps_stream = np.array(vis_fps_stream)
                        np_fps_stream = np_fps_stream[:,:2]    # take frames, seconds. drop unixtime.
                        np_fps = np.sum(np_fps_stream, axis=0) # [total_frames, total_seconds] duaring fps_stream_length

                        """
                        insert local values to shared variables
                        """
                        MPVariable.vis_fps_frames.value = int(np_fps[0]) # for console output
                        MPVariable.vis_fps.value = np_fps[0]/np_fps[1]   # for visualize and console
                        if MAX_VIS_FPS <= 0:
                            MPVariable.vis_skip_rate.value = 0
                        else:
                            rate = MPVariable.fps.value/MAX_VIS_FPS
                            MPVariable.vis_skip_rate.value = rate
                    else:
                        MPVariable.vis_fps_frames.value = 0
                        MPVariable.vis_fps.value = 0
                        MPVariable.vis_skip_rate.value = 0
                    previos_work_time = now_time
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            MPVariable.running.value = False
            if DEBUG_MODE:
                print("min_snapshot_fps:{:.1f}".format(min_snapshot_fps))
                print("{}".format(min_snapshot_list))
                print("max_snapshot_fps:{:.1f}".format(max_snapshot_fps))
                print("{}".format(max_snapshot_list))

        return


    def process_fps_console(self):
        """
        print fps by fps_interval sec.
        """
        logging.debug("enter")
        FPS_INTERVAL         = self.cfg['fps_interval']
        DEBUG_MODE           = self.cfg['debug_mode']
        SPLIT_MODEL          = self.cfg['split_model']

        sleep_interval = 1.0 / 50.0 # Wakeup and time checks N times per sec.

        try:
            while MPVariable.running.value and MPVariable.frame_counter.value == 0:
                time.sleep(sleep_interval)

            previos_work_time = time.time()
            while MPVariable.running.value:
                time.sleep(sleep_interval)
                now_time = time.time()
                if now_time >= previos_work_time + FPS_INTERVAL:
                    """
                    FPS console by fps_interval
                    """
                    frames = MPVariable.fps_frames.value
                    seconds = MPVariable.fps_seconds.value
                    if frames == 0:
                        total = 0
                        cap = 0
                        worker = 0
                        gpu = 0
                        cpu = 0
                        lost = 0
                    else:
                        total = MPVariable.total_proc_time.value/frames
                        cap = MPVariable.cap_proc_time.value/frames
                        worker = MPVariable.gpu_proc_time.value/frames
                        gpu = MPVariable.gpu_proc_time.value/frames
                        cpu = MPVariable.cpu_proc_time.value/frames
                        lost = MPVariable.lost_proc_time.value/frames
                    print("FPS:{: ^5.1f} Frames:{: ^3} Seconds:{: ^10.5f} | 1FRAME total:{: ^10.5f} cap:{: ^10.5f} worker:{: ^10.5f} gpu:{: ^10.5f} cpu:{: ^10.5f} lost:{: ^10.5f} send:{: ^10.5f} | VFPS:{: ^5.1f} VFrames:{: ^3} VDrops:{: ^3}"
                          .format(MPVariable.fps.value, MPVariable.fps_frames.value, MPVariable.fps_seconds.value,
                                  total,
                                  cap,
                                  worker,
                                  gpu,
                                  cpu,
                                  lost,
                                  MPVariable.send_proc_time.value,
                                  MPVariable.vis_fps.value, MPVariable.vis_fps_frames.value, MPVariable.vis_drop_frames.value))
                    MPVariable.cap_proc_time.value = 0
                    MPVariable.worker_proc_time.value = 0
                    MPVariable.gpu_proc_time.value = 0
                    MPVariable.cpu_proc_time.value = 0
                    MPVariable.lost_proc_time.value = 0
                    MPVariable.total_proc_time.value = 0
                    MPVariable.vis_proc_time.value = 0
                    MPVariable.vis_drop_frames.value = 0
                    previos_work_time = now_time
        except KeyboardInterrupt:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            MPVariable.running.value = False
            print("Time to first image:{}".format(MPVariable.first_complete_time.value))
        return

