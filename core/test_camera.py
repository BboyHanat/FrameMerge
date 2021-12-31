import cv2, time
import multiprocessing as mp
from multiprocessing import Value
import numpy as np


stop_flag = Value('b', False)


def image_put(q, stream_url):
    cap = cv2.VideoCapture(stream_url)
    while not stop_flag.value:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(queue_list, camera_ip_l):

    frames = [q.get() for q in queue_list]
    if len(frames) != len(camera_ip_l):
        return None
    return frames


def run_multi_stream(stream_urls):

    queues = [mp.Queue(maxsize=4) for _ in stream_urls]
    processes = list()
    for queue, url in zip(queues, stream_urls):
        processes.append(mp.Process(target=image_put, args=(queue, url)))

    for process in processes:
        process.daemon = True
        process.start()
    return queues, processes


if __name__ == '__main__':
    run_multi_stream()
