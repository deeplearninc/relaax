import numpy as np


class RingBuffer2D:
    """ A 2D ring buffer using numpy arrays """
    def __init__(self, element_size, buffer_size):
        self.data = np.zeros((buffer_size, element_size), dtype=np.float32)

    def extend(self, array_to_add):
        """ Adds array of element_size to buffer's end removing the beginning """
        self.data = np.vstack((self.data[1:, :], array_to_add))

    def get_sum(self):
        """ Returns the sum of elements within the rows of last half of the buffer """
        _, second = np.split(self.data, 2)
        return np.sum(second, axis=0)

    def get_diff(self, part):
        """ Returns the difference between the two halfs of data """
        first, second = np.split(self.data, 2)
        res = second - first
        to_grub = res.shape[0] - part
        grabbed_res = res[to_grub:, :]
        return grabbed_res

    def replace_second_half(self, array_to_replace):
        """ Replaces the buffer's hals from beginning by new array """
        first, _ = np.split(self.data, 2)
        self.data = np.vstack((first, array_to_replace))

    def reset(self):
        """ Resets buffer's data -> sets all elements to zeros """
        self.data.fill(0)
