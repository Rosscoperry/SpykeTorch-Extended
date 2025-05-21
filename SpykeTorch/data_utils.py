import math
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = "1"
import time

import numpy as np
import pandas as pd
import sys
from random import shuffle, seed
import torch
from matplotlib import pyplot as plt
from os import path
# from aermanager.parsers import parse_header_from_file, parse_dvs_ibm
from queue import Queue
from threading import Thread, Semaphore, BoundedSemaphore
import multiprocessing as mp
import threading as th
# from aermanager.dataset_generator import dataset_content_generator

# sys.path.append(r"C:\Users\ixb20175\Desktop\Projects")

import event_data_load.eventvision as ev
from expelliarmus import Wizard


class DataFeed:
    def __init__(self):
        ...

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PropheseeDataFeed(DataFeed):
    def __init__(self, dataset_path, time_window=10000, accepted_classes=None, image_size=(1280, 720)):
        super().__init__()
        self.dataset_path = dataset_path
        self.tw = time_window
        self.wizard = Wizard(encoding="evt3", time_window=time_window)
        self.file_list = []
        for root, subdirs, files in os.walk(dataset_path):
            for cls in subdirs:
                if accepted_classes is not None and cls not in accepted_classes:
                    continue
                self.file_list += [os.path.join(dataset_path, cls, fname) for fname in
                                   os.listdir(os.path.join(dataset_path, cls))]
        self._reset_files_iter()

    def _reset_files_iter(self):
        shuffle(self.file_list)
        self.iter = iter(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __iter__(self):
        ### Implementation 1: infinitelyyield time windows and notify if a new file is being processed using a flag.
        # while True:
        #     try:
        #         fpath = next(self.iter)
        #     except StopIteration as e:
        #         self._reset_files_iter()
        #         raise e
        #     cls = fpath.split(os.sep)[-2]
        #     self.wizard.set_file(fpath)
        #     new_file = True
        #     for tw in self.wizard.read_time_window():
        #         yield tw, cls, new_file
        #         new_file = False
        #     new_file = True
        ### Implementation 2: yield a generator for each new file, hence expect other code outside to iterate through it
        while True:
            try:
                fpath = next(self.iter)
            except StopIteration as e:
                self._reset_files_iter()
                return
            cls = fpath.split(os.sep)[-2]
            self.wizard.set_file(fpath)
            yield self.wizard.read_time_window(), cls, fpath


def events_to_frame(frame_size=(1280, 720), area_of_interest=None):
    """

    :param frame_size: (width, height)
    :param area_of_interest: ((width_left, width_right), (height_top, height_bottom))
    :return: 2-channel image with events
    """

    aoi = area_of_interest
    if aoi is not None:
        frame_size = (aoi[0][1] - aoi[0][0], aoi[1][1] - aoi[1][0])
    def _ev2frame(event_list):
        image = np.zeros((2,) + frame_size)  # adds a channel dimension of size 2 (negative and positive events)
        for ev in event_list:
            if True or (aoi is not None and aoi[0][1] >= ev[2] >= aoi[0][0] and aoi[1][1] >= ev[1] >= aoi[1][0]):
                image[ev[3], ev[2], ev[1]] = 1
        return image

    return _ev2frame


class NeuromorphicDataFeed(DataFeed):
    """
    Iterable source of NeuroMorphic (NM) data.  AER data (events data) is read and all the spikes happened
    within timestep t0 and t1 are encoded into a height x width numpy matrix.
    If more than one spike happened at a certain location within this time frame, it is lost (only one gets to be
    encoded, no accumulation).
    Currently works with N-MNIST only.
    """

    # only works with n-mnist for now

    def __init__(self, path, dt, image_size, to_torch_tensor=True, accepted_classes=None, roi=None,
                 excitatory_only=True, seed=None):
        super().__init__()
        self.excitatory_only = excitatory_only
        self.path = path
        self.dt = dt
        self.image_size = image_size
        self.spike_value = 1  # / dt
        self.n_classes = 0
        self.reference_time = 0
        self.accum_time = 0
        self.roi = roi  # Region of interest in the form [[start_x, start_y], [end_x, end_y]]
        self.accepted_classes = accepted_classes
        self.to_torch_tensor = to_torch_tensor
        self.seed = seed
        self._read_dataset()
        self.f = None

    def _read_dataset(self):
        self.data_dict = {}
        self.data_array = []
        for root, subdirs, _ in os.walk(self.path):
            for label in subdirs:
                if self.accepted_classes is not None and label not in self.accepted_classes:
                    continue
                self.n_classes += 1
                # if label != "1": continue
                files = os.listdir(os.path.join(self.path, label))
                self.data_dict[label] = files
                self.data_array += [label + os.path.sep + file for file in files]
        self.data_array = np.array(self.data_array)
        self.data_array_bak = self.data_array.copy()
        self._reset_files_iter()

    def reset(self):
        self._reset_files_iter()

    def _reset_files_iter(self):
        if self.seed is not None:
            seed(self.seed)
        idx = np.arange(len(self.data_array))
        shuffle(idx)
        self.data_array = self.data_array_bak[idx]
        self.length = len(self.data_array)
        self.data_iter = iter(self.data_array)
        self.reference_time = 0
        self.accum_time = 0
        self.current_file = None
        self.events_iter = None
        self.current_event = None

    def __next__(self):
        spikes = self(self.accum_time)
        self.accum_time += self.dt

        # formatting into pytorch-like tensor
        if self.excitatory_only:
            spikes = spikes.reshape(self.image_size)
            spikes = np.expand_dims(spikes, axis=0)
            spikes = np.expand_dims(spikes, axis=0)
        else:
            spikes = spikes.reshape((1, 2) + self.image_size)
        if self.to_torch_tensor:
            spikes = torch.Tensor(spikes)
        return spikes, self.current_file

    def __iter__(self):
        return self

    def __call__(self, t):
        if self.current_file is None:
            try:
                if self.f is None:
                    self.current_file = next(self.data_iter)
                    # self.f = self.current_file
                else:
                    self.current_file = self.f
                # acceptables = ["{}".format(k) for k in range(2)]
                # while self.current_file.split("\\")[0] not in acceptables:
                #     self.current_file = next(self.data_iter)
                # print("Feeding file : {}".format(self.current_file))
            except StopIteration as e:
                # end of dataset
                self._reset_files_iter()
                raise e

        if self.events_iter is None:
            self.events_iter = iter(self._read_events(self.current_file))

        if self.current_event is None:
            self.current_event = next(self.events_iter)
            self.reference_time = t

        if self.excitatory_only:
            spikes = np.zeros(self.image_size)
        else:
            spikes = np.zeros((2,) + self.image_size)
        while self.current_event.ts < (t - self.reference_time) * 1e6:
            if self.excitatory_only or self.current_event.p:
                spikes[0, self.current_event.y, self.current_event.x] = self.spike_value
            else:
                spikes[1, self.current_event.y, self.current_event.x] = self.spike_value

            try:
                self.current_event = next(self.events_iter)
            except StopIteration:
                # end of events for the current image
                self.current_event = None
                self.current_file = None
                self.events_iter = None
                break
        return spikes.reshape(-1)

    def _read_events(self, filename):
        events = ev.read_dataset(os.path.join(self.path, filename))
        if self.roi:
            events.data = events.extract_roi(self.roi[0], self.roi[1])
            events.data.x -= self.roi[0, 0]
            events.data.y -= self.roi[0, 1]
        return events.sort_order()

    def __len__(self):
        return self.length

    def label_gen(self):

        def _label_gen(t):
            label_spikes = np.zeros(self.n_classes)
            if self.current_file is not None:
                current_label = int(self.current_file.split(os.path.sep)[0])
                label_spikes[current_label] = self.spike_value
            return label_spikes

        return _label_gen


class MPNeuromorphicDataFeed(DataFeed):
    def __init__(self, path, dt, image_size, to_torch_tensor=True, accepted_classes=None, roi=None,
                 excitatory_only=True, seed=None, num_workers=4, threading=False):
        super().__init__()
        self.excitatory_only = excitatory_only
        self.path = path
        self.dt = dt
        self.image_size = image_size
        self.spike_value = 1  # / dt
        self.n_classes = 0
        self.reference_time = 0
        self.accum_time = 0
        self.processed_files = 0
        self.roi = roi  # Region of interest
        self.accepted_classes = accepted_classes
        self.to_torch_tensor = to_torch_tensor
        self.seed = seed
        self.num_workers = num_workers
        self.data_queue = mp.Queue()
        self.result_queue = mp.Queue(maxsize=20)
        self.workers = []
        self.threading = threading
        self._read_dataset()
        # self._start_workers()

    def _read_dataset(self):
        self.data_dict = {}
        self.data_array = []
        for root, subdirs, _ in os.walk(self.path):
            for label in subdirs:
                if self.accepted_classes is not None and label not in self.accepted_classes:
                    continue
                self.n_classes += 1
                files = os.listdir(os.path.join(root, label))
                self.data_dict[label] = files
                self.data_array += [os.path.join(self.path, label, file) for file in files]
        self.data_array = np.array(self.data_array)
        self.data_array_bak = self.data_array.copy()
        self._reset_files_iter()

    @staticmethod
    def _read_events(fullpath, roi):
        events = ev.read_dataset(os.path.join(fullpath))
        if roi:
            events.data = events.extract_roi(roi[0], roi[1])
            events.data.x -= roi[0, 0]
            events.data.y -= roi[0, 1]
        return events.sort_order()

    def _reset_files_iter(self):
        if self.seed is not None:
            seed(self.seed)
        np.random.shuffle(self.data_array)
        self.length = len(self.data_array)
        self.processed_files = 0
        for data in self.data_array:
            self.data_queue.put(data)
        if len(self.workers) == 0:
            self._start_workers()

    def _start_workers(self):
        if self.threading:
            for _ in range(self.num_workers):
                worker = th.Thread(target=self.data_loader_worker, args=(self.data_queue, self.result_queue,
                                                                         self.excitatory_only, self.dt,
                                                                         self.image_size,
                                                                         self.roi))
                worker.start()
                self.workers.append(worker)
        else:
            for _ in range(self.num_workers):
                worker = mp.Process(target=self.data_loader_worker, args=(self.data_queue, self.result_queue,
                                                                          self.excitatory_only, self.dt,
                                                                          self.image_size,
                                                                          self.roi))
                worker.start()
                self.workers.append(worker)

    @staticmethod
    def data_loader_worker(data_queue, result_queue, excitatory_only, dt, image_size, roi):
        while True:
            data_path = data_queue.get()
            if data_path is None:
                break

            result = MPNeuromorphicDataFeed.process_data(data_path, excitatory_only, dt, image_size, roi)
            result_queue.put(result)

    @staticmethod
    def process_data(data_path, excitatory_only, dt, image_size, roi):
        events_iter = iter(MPNeuromorphicDataFeed._read_events(data_path, roi))

        current_event = next(events_iter)
        reference_time = 0
        t = dt

        batched = []
        while True:
            if excitatory_only:
                spikes = np.zeros(image_size)
            else:
                spikes = np.zeros((2,) + image_size)
            try:
                while current_event.ts < (t - reference_time) * 1e6:
                    if excitatory_only or current_event.p:
                        spikes[0, current_event.y, current_event.x] = 1
                    else:
                        spikes[1, current_event.y, current_event.x] = 1

                    current_event = next(events_iter)
                batched.append(spikes)
                t += dt
            except StopIteration:
                batched.append(spikes)
                break
        batched = torch.tensor(np.array(batched))
        return batched

    def __next__(self):
        if self.processed_files >= self.length:
            # self.close()
            self._reset_files_iter()
            raise StopIteration("No more data available.")
        self.processed_files += 1
        return self.result_queue.get(), self.data_array[self.processed_files - 1]

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __del__(self):
        self.close()

    def close(self):
        for _ in range(self.num_workers):
            self.data_queue.put(None)
        for worker in self.workers:
            worker.join()


class GestureEvent:

    def __init__(self, event_info, from_bytes=False):
        if from_bytes:
            event_data = int.from_bytes(event_info[0], byteorder="big")  # may be the cause of some problems.. ?
            event_timestamp = int.from_bytes(event_info[1], byteorder="big")
            self.x = (event_data >> 17) & 0x00001FFF
            self.y = (event_data >> 2) & 0x00001FFF
            self.p = (event_data >> 1) & 0x00000001
            self.ts = event_timestamp
            self.ts_ms = self.ts / 1e3
        else:
            self.x = event_info[0]
            self.y = event_info[1]
            self.ts = event_info[2]
            self.p = event_info[3]


class GestureEventSequence:

    def __init__(self, filename, start_byte=None, end_byte=None, label=None):
        self.label = label
        self.end_byte = end_byte
        self.start_byte = start_byte
        self.filename = filename
        self.events = []
        self.event_indexes = []
        self.event_count = 0
        self.next_event_index = 0
        self.start_ts = None
        self.ref_time = 0
        self.max_x = self.max_y = 0
        self.min_x = self.min_y = sys.maxsize

    def set_bulk_events(self, events):
        self.events = events
        self.event_count = len(events)
        self.event_indexes = list(range(self.event_count))
        self.start_ts = events[0].ts
        self.ref_time = self.start_ts

    def add_event(self, event):
        self.events.append(event)
        self.event_indexes.append(self.event_count)
        self.event_count += 1
        if self.start_ts is None:
            self.start_ts = event.ts
            self.ref_time = event.ts

    def reset(self):
        self.ref_time = self.start_ts
        self.next_event_index = 0

    def __call__(self, t):
        """

        Args:
            t:

        Returns: Tuple: (list of events within that timeframe, is_last). If list is empty it means no events,
        if return is None it means sequence is finished.

        """
        # if not self.called_once:
        #     self.called_once = True
        #     print(f"x: [{self.min_x}-{self.max_x}] | y: [{self.min_y}-{self.max_y}]")
        if self.next_event_index >= self.event_count:
            return None, False
        events = []
        self.ref_time += t
        while self.next_event_index < self.event_count and self.events[self.next_event_index].ts <= self.ref_time:
            events.append(self.events[self.next_event_index])
            self.next_event_index += 1
        return events, self.next_event_index >= self.event_count


class GestureEventSequenceCollection:

    def __init__(self, filename, mapping_file, dt, accepted_classes=None, verbose=False):
        self.accepted_classes = accepted_classes
        self.dt = dt
        self.filename = os.path.expanduser(filename)
        self.data_version, self.data_start = parse_header_from_file(self.filename)
        self.class_mapping = pd.read_csv(os.path.expanduser(mapping_file))  # [:-1]  # drop the "other" class
        self.indexes = list(range(len(self.class_mapping)))  # - 1))
        self.sequences = []
        self.accepted_sequences = []
        self.accum_time = 0
        self.iter_indexes = self.iter_indexes_bak = None
        self._i = 0
        self.n = 0
        self.am_i_over = False
        self.verbose = verbose
        self._read_events()

    def _read_events(self):
        label = 0
        if self.verbose:
            print("\nReading events contained in {}".format(self.filename))

        _, events = parse_dvs_ibm(self.filename)

        # start = time.time()
        for i in range(len(self.class_mapping)):
            # we start from 0, dataset from 1
            label = self.class_mapping["class"][i] - 1
            if self.accepted_classes is not None and label not in self.accepted_classes:
                continue
            t_start = self.class_mapping["startTime_usec"][i]
            t_end = self.class_mapping["endTime_usec"][i]
            mask = np.logical_and(events["t"] >= t_start, events["t"] < t_end)
            class_events = list(map(lambda ev: GestureEvent(ev), events[:][mask]))
            current_sequence = GestureEventSequence(self.filename, label=label)
            current_sequence.set_bulk_events(class_events)
            self.sequences.append(current_sequence)

        # print(f"Time taken {time.time()-start}")

        self.n = len(self.sequences)
        self.iter_indexes = list(range(len(self.sequences)))
        self.iter_indexes_bak = self.iter_indexes.copy()

        shuffle(self.iter_indexes)
        if self.verbose:
            print("Done.")

    def reset(self):
        self.am_i_over = False
        self.iter_indexes = self.iter_indexes_bak.copy()
        shuffle(self.iter_indexes)
        self._i = 0
        if self.accepted_classes is not None:
            for s in self.accepted_sequences:  # type: GestureEventSequence
                s.reset()
        else:
            for s in self.sequences:  # type: GestureEventSequence
                s.reset()

    def __next__(self):
        return self()

    def __iter__(self):
        return self

    def __call__(self):

        # while self.accepted_classes is not None and \
        #         self._i < self.n and \
        #         self.sequences[self.iter_indexes[self._i]].label not in self.accepted_classes:
        #     self._i += 1
        if self.am_i_over:
            raise StopIteration

        events, is_last = self.sequences[self.iter_indexes[self._i]](
            self.dt)  # automatically keeps an internal reference time that is increased
        target = self.sequences[self.iter_indexes[self._i]].label
        if is_last:
            target = None  # this is to let the outmost logic that the current sequence has ended
            self._i += 1
        if self._i >= self.n:
            self.am_i_over = True
        return events, target, is_last, self.am_i_over


class DVSGesturesReader(DataFeed):

    def __init__(self, parent_folder, dt, image_size=(128, 128), accepted_classes=None, spike_amplitude=1,
                 feed_test=False, seed=None):
        super().__init__()
        self.feed_test = feed_test
        self.parent_folder = parent_folder
        self.dt = dt
        self.dt_s = dt / 1e6
        self.image_size = image_size
        self.accepted_classes = accepted_classes
        self.accum_time = 0
        self.spike_value = spike_amplitude / self.dt_s
        self.seed = seed

        print("Reading train and test set files from\n{} \nand\n{}".format(
            path.join(parent_folder, "trials_to_train.txt"),
            path.join(parent_folder, "trials_to_test.txt")))

        with open(path.join(parent_folder, "trials_to_train.txt"), "r") as f:
            self.train_set = list(map(lambda s: s.split(".")[0],
                                      f.read().splitlines()))
            self.train_indexes = list(range(len(self.train_set)))
            self.train_index_bak = self.train_indexes.copy()
            shuffle(self.train_indexes)
        with open(path.join(parent_folder, "trials_to_test.txt"), "r") as f:
            self.test_set = list(map(lambda s: s.split(".")[0],
                                     f.read().splitlines()))
            self.test_indexes = list(range(len(self.test_set)))
            self.test_index_bak = self.test_indexes.copy()
            shuffle(self.test_indexes)
        print("Done.")
        self._train_idx = 0
        self._test_idx = 0
        self.train_n = len(self.train_set)
        self.test_n = len(self.test_set)
        self.current_collection = None
        self.feed = None
        self.init_feed()

    def clear_resources(self):
        self.feed.clear()

    def reset(self):
        self._reset_files_iter()

    def _reset_files_iter(self):
        self.current_collection = None
        self._train_idx = 0
        self._test_idx = 0
        self.train_indexes = self.train_index_bak.copy()
        self.test_indexes = self.test_index_bak.copy()
        if self.seed is not None:
            seed(self.seed)
        shuffle(self.train_indexes)
        shuffle(self.test_indexes)
        self.clear_resources()
        self.init_feed()

    def toggle_test_data(self):
        self.feed_test = not self.feed_test
        print(f"Setting test data feed to {self.feed_test}.")
        self.reset()

    def __len__(self):
        n = self.train_n if not self.feed_test else self.test_n
        if self.accepted_classes is not None:
            return n * len(self.accepted_classes)
        return n * 10

    def __next__(self):
        for data in self.feed:
            event_map = np.zeros(self.image_size)
            events, target, is_last, is_collection_over = data
            for event in events:
                event_map[event.x, event.y] = self.spike_value

            event_map = np.expand_dims(event_map, axis=0)
            event_map = np.expand_dims(event_map, axis=0)
            yield torch.Tensor(event_map), str(target) if not is_last else None
        # spikes, target = next(self.yield_it())  # self(self.dt)
        # spikes = torch.Tensor(spikes)
        # yield spikes, target

    def __iter__(self):
        return self.__next__()

    def init_feed(self):
        if not self.feed_test:
            self.feed = SequenceCollectionGenerator(self.parent_folder, self.train_set, self.train_indexes,
                                                    self.dt, self.accepted_classes)
        else:
            self.feed = SequenceCollectionGenerator(self.parent_folder, self.test_set, self.test_indexes,
                                                    self.dt, self.accepted_classes)

    def yield_it(self):
        for data in self.feed:
            event_map = np.zeros(self.image_size)
            events, target, is_last, is_collection_over = data
            for event in events:
                event_map[event.x, event.y] = self.spike_value

            yield event_map, str(target) if not is_last else None

    # Deprecated #
    def __call__(self, t):
        if self._train_idx >= self.train_n:
            raise StopIteration

        if self.current_collection is None:
            filename = self.train_set[self.train_indexes[self._train_idx]]
            self.current_collection = GestureEventSequenceCollection(path.join(self.parent_folder, filename + ".aedat"),
                                                                     path.join(self.parent_folder,
                                                                               filename + "_labels.csv"),
                                                                     self.dt)
            self._train_idx += 1
        event_map = np.zeros(self.image_size)
        events, target, is_last, is_collection_over = self.current_collection()
        if is_collection_over:
            self.current_collection = None
        for event in events:
            event_map[event.x, event.y] = self.spike_value

        return event_map, str(target) if not is_last else None


def print_nm_image(data_loader, n_ts=10, color_map=None, savepath=None):
    """

    Args:
        data_loader:
        n_ts: number of time steps to sample.
        color_map: an array of shape (3, 3) where each row encodes the RGB value for overlapping, positive and negative
        areas.

    Returns:

    """
    import matplotlib.pyplot as plt
    if color_map is None:
        color_map = [
            [220, 220, 0],
            [70, 220, 70],
            [220, 70, 70]
        ]
    spikes = None
    was_event = None
    i = 0
    data_loader._reset_files_iter()
    for spike_wave, file in data_loader:
        if spikes is None:
            spikes = spike_wave
            was_event = torch.full(spike_wave.size(), False)
        else:
            spikes += spike_wave
            was_event[spike_wave != 0] = True
        i += 1
        if i >= n_ts:
            break

    image_s = spikes.numpy()
    image_s = image_s.squeeze()
    image_s = image_s.squeeze()
    image_s = np.moveaxis(image_s, [0, 1], [-1, -2])

    we = was_event.numpy()
    we = we.squeeze()
    we = we.squeeze()
    we = np.moveaxis(we, [0, 1], [-1, -2])

    img = np.expand_dims(image_s, axis=-1)
    img = np.append(img, np.expand_dims(image_s, axis=-1), axis=-1)
    img = np.append(img, np.expand_dims(image_s, axis=-1), axis=-1)

    img[we] = color_map[0]
    img[image_s > 0] = color_map[1]
    img[image_s < 0] = color_map[2]

    img = img.astype(int)
    plt.imshow(img)
    plt.axis("off")
    if savepath:
        plt.savefig(savepath)
    plt.show()
    return img


def getDatasetStats(dataset: DataFeed):
    counters = {}
    count = 0
    c = 0
    t = None
    avg_spikes_per_file = []
    spikes_n = 0
    ts = 0
    spikes_throughout_ts = {}
    for data in dataset:
        spike_map, file = data
        ts += 1
        if file is not None and t is None:
            t = file.split("\\")[0]

        count = torch.count_nonzero(spike_map).item()
        spikes_n += count
        try:
            counters[t]
        except:
            counters[t] = {}
        try:
            spikes_throughout_ts[t]
        except:
            spikes_throughout_ts[t] = {}
        try:
            spikes_throughout_ts[t][ts]
            spikes_throughout_ts[t][ts] += count
        except:
            spikes_throughout_ts[t][ts] = count

        try:
            counters[t][count] += 1
        except:
            counters[t][count] = 1

        if file is None:
            t = None
            c += 1
            avg_spikes_per_file.append(spikes_n / ts)
            spikes_n = ts = 0
        log_string = "\r{:.2f}% completed."
        print(log_string.format(c / len(dataset) * 100), end="")

    return counters, avg_spikes_per_file, spikes_throughout_ts


def get_collection(id: int, queue: Queue, parent_folder, paths: list, dt, accepted_classes, paths_q=None):
    if paths_q is not None:
        empty = False
        while not empty:
            try:
                filename = paths_q.get(timeout=2)
                collection = GestureEventSequenceCollection(path.join(parent_folder, filename + ".aedat"),
                                                            path.join(parent_folder, filename + "_labels.csv"),
                                                            dt,
                                                            accepted_classes=accepted_classes)
                queue.put(collection)
            except:
                # check if actually empty or was just slow
                empty = paths_q.empty()

    else:
        for filename in paths:
            # print(f"[Th. {id}] Reading {filename}.")
            collection = GestureEventSequenceCollection(path.join(parent_folder, filename + ".aedat"),
                                                        path.join(parent_folder, filename + "_labels.csv"),
                                                        dt,
                                                        accepted_classes=accepted_classes)
            queue.put(collection)
            # print(f"\n[Pr. {id}] Done and queued {filename}.")


class SequenceCollectionGenerator:

    def __init__(self, parent_folder, data_paths, indexes, dt, accepted_classes, n_proc=2, verbose=True):
        self.verbose = verbose
        self.accepted_classes = accepted_classes
        self.dt = dt
        self.parent_folder = parent_folder
        self.indexes = indexes
        self.paths = data_paths
        self.n_proc = n_proc
        self.processes = []
        self.paths_copy = data_paths.copy()
        self.paths_q = mp.Queue()
        for p in self.paths_copy:
            self.paths_q.put(p)
        self.n = len(self.paths_copy)
        self.collection_queue = mp.Queue(maxsize=5)
        dn = math.ceil(self.n / self.n_proc)
        for i in range(self.n_proc):
            sliced = self.paths_copy[i * dn: i * dn + dn]
            if len(sliced) > 0:
                p = mp.Process(target=get_collection,
                               args=(i, self.collection_queue, self.parent_folder,
                                     sliced, self.dt, self.accepted_classes, self.paths_q),
                               name=f"CollectionGetter{i}")
                self.processes.append(p)
                p.start()
            else:
                break

    def reset(self):
        raise NotImplementedError

    def __next__(self):
        # todo loop until threads/proc are done? so you dont care about how many there are, just let them handle it.
        for i in range(len(self.paths)):
            collection = self.collection_queue.get()
            for data in collection:
                yield data
        # for t in self.processes:
        #     t.join()

    def __iter__(self):
        return next(self)

    def clear(self):
        for p in self.processes:
            if p.is_alive():
                if self.verbose:
                    print("Terminating process {}".format(p.name))
                p.terminate()


if __name__ == "__main__":
    dt = 0.05
    train_loader = DVSGesturesReader(r"C:\Users\ixb20175\Desktop\Datasets\DVS  Gesture dataset\DvsGesture",
                                     dt=dt * 1e6, accepted_classes=[0, 1])

    # train_loader = NeuromorphicDataFeed(r"C:\Users\ixb20175\Desktop\Datasets\N-MNIST\Train",
    #                                     dt=dt, image_size=(28, 28), seed=15)

    stats, avg_spikes, spike_through_time = getDatasetStats(train_loader)
    for k in spike_through_time.keys():
        spikes = spike_through_time[k]
        plt.bar(np.arange(len(spikes)), spikes.values())
        plt.title(f"Events distribution over time for class {k}")
        plt.xlabel("Time (s)")
        plt.ylabel("Event count (10\u2074)")
        ticks, _ = plt.yticks()
        plt.yticks(ticks, [int(x / 1e4) if x > 0 else x for x in ticks])
        plt.xticks(ticks=np.arange(1, len(spikes), step=20),
                   labels=[round(n * dt, 3) for n in np.arange(1, len(spikes.keys()), step=20)],
                   rotation=70)
        plt.tight_layout()
        plt.savefig(f"../gestures_stats/event_distribution_class_{k}.png")
        plt.show()

    ll = [d.values for d in spike_through_time.values()]
    np_matrix = pd.DataFrame(ll).fillna(0).values.astype(int)
    plt.figure(figsize=(20, 10))
    plt.boxplot(np_matrix, medianprops=dict(linewidth=3))
    plt.title(f"Inter-class events distribution over time.")
    plt.xlabel("Time (s)")
    plt.ylabel("Event count (10\u2074)")
    ticks, _ = plt.yticks()
    plt.yticks(ticks, [int(x / 1e4) if x >= 0 else x for x in ticks])
    plt.xticks(ticks=np.arange(1, len(np_matrix[0]), step=20),
               labels=[round(n * dt, 3) for n in np.arange(1, len(np_matrix[0]), step=20)],
               rotation=70)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"../gestures_stats/inter_class_events_distribution.png")
    plt.show()
    # print(stats)
