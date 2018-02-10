import numpy as np
import math

from abc import abstractmethod


class Gradient:
    """
    Base class for building Gradients - segmented data with associated colours
    """

    def __init__(self, base_colour: tuple=(0, 0, 0), peak_colour: tuple=(255, 0, 0)):

        self._data = None
        self._x_axis = None
        self._mappings = None
        self._base_colour = base_colour
        self._peak_colour = peak_colour

    def _set_data(self, data: np.ndarray, x: np.ndarray=None):

        self._data = data
        if x is None:
            self._x_axis = np.linspace(0, self._data.shape[0], self._data.shape[0])
        elif x.shape[0] != self._data.shape[0]:
            raise AttributeError("data and axis must have the same shape")
        else:
            self._x_axis = x

    def __get_colour(self, x):
        """
        for a given data point, lookup it's colour
        :param x: data point
        :return:  its colour
        """
        if self._mappings is not None:
            colour = self._base_colour
            for thershold, c in sorted(self._mappings.items()):
                if x >= thershold:
                    colour = c
            return colour
        return self._base_colour

    def _build_plot(self):
        """
        Prepares data for building a plot
        :return:
        """
        return [([self._x_axis[i - 1], self._x_axis[i]],
                 [self._data[i - 1], self._data[i]],
                 self.__get_colour(self._data[i - 1])) for i in range(1, self._data.shape[0])]

    def _build_scatter(self):
        """
        Prepares data for building a scatter plot
        :return: a list of (value, plot colour) tuples
        """
        return [(x, y, self.__get_colour(y)) for x, y in zip(self._x_axis, self._data)]

    @abstractmethod
    def build_plot(self, data: np.ndarray, x_axis: np.ndarray=None):
        """

        :param data:
        :param x_axis:
        :return:
        """

    @abstractmethod
    def build_scatter(self, data: np.ndarray, x_axis: np.ndarray = None):
        """

        :param data:
        :param x_axis:
        :return:
        """


class SegmentedGradient(Gradient):
    """
    Explicitly define which regions of the data are set to which colours
    """

    def __init__(self, base_colour: tuple=(0, 0, 0)):

        super(SegmentedGradient, self).__init__(base_colour=base_colour)

    def add_threshold(self, threshold: float, colour: tuple):
        """
        Apply colour to data above this threshold
        :param threshold:   threshold value
        :param colour:      colour to apply
        """
        self._mappings = self._mappings if self._mappings is not None else {}
        self._mappings[threshold] = colour
        return self

    def build_plot(self, data: np.ndarray, x_axis: np.ndarray=None):

        self._set_data(data, x_axis)
        return self._build_plot()

    def build_scatter(self, data: np.ndarray, x_axis: np.ndarray = None):

        self._set_data(data, x_axis)
        return self._build_scatter()


class ContinuousGradient(Gradient):
    """
    Continuous gradient - colour is interpolated to by default or set explicitly to a fixed number of intervals
    """

    def __init__(self, base_colour: tuple = (0, 0, 0), peak_colour: tuple=(255, 0, 0)):

        super(ContinuousGradient, self).__init__(base_colour=base_colour, peak_colour=peak_colour)
        self._segments = None

    def set_segments(self, num_segments):
        """
        set the number of coloured segments explicitly
        :param num_segments: number of evenly spaced colour segments
        """
        self._segments = num_segments

    def _build_mappings(self):
        """
        Sets the mappings given the number of segments set
        """
        self._mappings = {}
        minn = np.min(self._data)
        maxx = np.max(self._data)
        for i in range(self._segments + 1):
            threshold = minn + (i * (maxx - minn)) / self._segments
            colour = tuple(int(math.floor(b + i * (p - b) / self._segments)) if p > b else
                           int(math.floor(p + i * (b - p) / self._segments))
                      for b, p in zip(self._base_colour, self._peak_colour))
            self._mappings[threshold] = colour

    def build_plot(self, data: np.ndarray, x_axis: np.ndarray=None):

        self._set_data(data, x_axis)
        self._segments = self._segments if self._segments is not None else data.shape[0]
        self._build_mappings()
        return self._build_plot()

    def build_scatter(self, data: np.ndarray, x_axis: np.ndarray = None):

        self._set_data(data, x_axis)
        self._segments = self._segments if self._segments is not None else data.shape[0]
        self._build_mappings()
        return self._build_scatter()
