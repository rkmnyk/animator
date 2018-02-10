import matplotlib.pyplot as plt
import numpy as np
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


class Animator():

    def __init__(self):
        """

        """


    @staticmethod
    def build_video_from_files(directory: str, filename: str, fps: int, size: tuple=None,
                         is_color: bool=True, format: str="XVID"):
        """
        Create a video from a folder of images, sorted by filename.
        By default, the video will have the size of the first image.
        It will resize every image to this size before adding them to the video.

        :param directory:   path to files
        :param filename:    name of video
        :param fps:         frames per second
        :param size:        video dimensions
        :param is_color:    colour video
        :param format:      see http://www.fourcc.org/codecs.php
        :return: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        """
        fourcc = VideoWriter_fourcc(*format)
        vid = None

        for file in sorted(os.listdir(directory)):
            if not os.path.exists(file):
                raise FileNotFoundError(file)
            image = imread(file)
            if vid is None:
                if size is None:
                    size = image.shape[1], image.shape[0]
                vid = VideoWriter(filename, fourcc, fps, size, is_color)
            try:
                if size[0] != image.shape[1] and size[1] != image.shape[0]:
                    image = resize(image, size)
            except AttributeError:
                print(file)
            vid.write(image)
        vid.release()
        return vid

    @staticmethod
    def build_video_from_data(data, fps, filename, secondary_labels=None):
        """

        :param data:                1d numpy array of data to animate
        :param fps:                 frames per second
        :param filename:            filename to write video to
        :param secondary_labels:
        :return:
        """

        if os.path.isfile(filename):
            os.remove(filename)

            # size and resolution settings (dpi = your approximate monitor dpi)
        dpi = 64
        h = 720
        w = 1280

        # colour and tier settings
        # tiers = [0.45, 0.8]
        tiers = [0.7, 0.8]
        tier_colours = ["k", "c", "b"]

        # label settings
        label_tolerance = tiers[-1]
        spacer = 5
        default_label = "Anomaly"
        a_props = dict(facecolor='black', shrink=0.01, width=1, headwidth=6)

        # box colouring presets
        box_colour_1 = [1, 153 / 255, 153 / 255]
        box_colour_2 = [153 / 255, 153 / 255, 102 / 255]

        # box colouring keys
        box_colours = {}
        box_colours[default_label] = box_colour_1
        if secondary_labels is not None:
            for lab in secondary_labels:
                box_colours[lab[1]] = np.random.rand(3)
                # box_colors[lab[1]] = box_colour_2

        # plot settings and styles
        lw = 5
        fs = 20
        y_label = "Anomaly Score"
        y_ticks = ([0, 0.25, 0.5, 0.75, 1])
        y_ticklables = y_ticks
        ymax = 2
        ymin = -2

        # image and video writers
        fourcc = VideoWriter_fourcc(*'XVID')
        size = w, h
        vid = VideoWriter(filename, fourcc, fps, size, True)
        fig = plt.figure(figsize=(w / dpi, h / dpi))


        for i in range(len(data)):

            # set the plot, the limits, ticks, axes etc
            ax = fig.add_subplot(111)
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks, fontsize=int(fs * 0.8))
            ax.set_ylabel(y_label, fontsize=fs)
            ax.set_xticks([])

            # get the correct slice of data for this iteration and pad accordingly
            if i < fps:
                frame_data = np.hstack((np.zeros(fps - i), data[:i]))
            else:
                frame_data = data[i - fps:i]

            # plot basic
            ax.plot(frame_data, linewidth=lw, color=tier_colours[0])

            # overlay for all plot colours
            """
            for j in range(len(tiers)):

                tier = np.where(frame_data >= tiers[j])[0]
                c = tier_colours[j + 1]
                if tier.shape[0] > 0:
                    splits = np.insert(np.where(np.diff(tier) > 1)[0], 0, -1)
                    if splits.shape[0] > 1:
                        plot_segments = [tier[splits[k] + 1:splits[k + 1] + 1] for k in range(splits.shape[0] - 1)]
                        plot_segments.append(tier[splits[-1] + 1:])
                        for plot_segment in plot_segments:
                            ax.plot(plot_segment, [frame_data[k] for k in plot_segment], linewidth=lw, color=c)
                    else:
                        ax.plot(tier, [frame_data[k] for k in tier], linewidth=lw, color=c)

            # threshold settings here
            if np.any(np.greater(frame_data, label_tolerance)):

                pos = [j for j in range(len(frame_data)) if frame_data[j] >= label_tolerance]
                labels = []

                if secondary_labels is None:
                    s = 0
                    c = 1
                    for c in range(1, len(pos)):
                        if pos[c] - pos[c - 1] != 1:
                            labels.append((default_label, pos[int((c - s) / 2)]))
                            s = c
                        elif c == len(pos) - 1:
                            labels.append((default_label, pos[int((c - s) / 2)]))
                else:

                    active_vals = []
                    active_labs = []
                    for l in secondary_labels:
                        if (i - 2 * fps) < l[0] and (i - fps) > l[0]:
                            active_vals.append(0)
                            active_labs.append(l[1])
                        elif (i - fps) <= l[0] and i >= l[0]:
                            active_vals.append(fps - (i - l[0]))
                            active_labs.append(l[1])
                        elif i < l[0] and (i + fps) > l[0]:
                            active_vals.append(fps - 1)
                            active_labs.append(l[1])
                    c = 1
                    s = 0
                    found = False
                    for c in range(1, len(pos)):
                        if pos[c] - pos[c - 1] == 1:
                            if pos[c - 1] in active_vals:
                                labels.append((active_labs[active_vals.index(pos[c - 1])], pos[c - 1]))
                                found = True
                            if c == len(pos) - 1:
                                if pos[c] in active_vals:
                                    labels.append((active_labs[active_vals.index(pos[c])], pos[c]))
                                else:
                                    if found is False:
                                        labels.append((default_label, pos[int((c - s) / 2)]))
                        else:
                            if found is False:
                                labels.append((default_label, pos[int((c - s) / 2)]))
                            else:
                                found = False
                            s = c

                            # print(pos, active_vals, active_labs, labels)

                for j, l in enumerate(labels):
                    ax.annotate(l[0], xy=(l[1], frame_data[l[1]]), xytext=(0.1 * fps + (j * spacer), 1.3),
                                arrowprops=a_props, fontsize=fs,
                                bbox={'facecolor': box_colours[l[0]], 'alpha': 0.5, 'pad': 5})
                                """
            fig.canvas.draw()

            image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            if size[0] != image.shape[1] and size[1] != image.shape[0]:
                image = resize(image, size)

            vid.write(image)
            fig.clear()
            ax.clear()

        vid.release()
        plt.close()
        return vid

