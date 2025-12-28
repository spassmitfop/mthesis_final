import math

import cv2
import numpy as np
import gymnasium as gym
from collections import deque

from matplotlib import pyplot as plt
from ocatari.ram.extract_ram_info import get_class_dict, get_max_objects


class Scaler():
    """
    Class for scaling an input representation.
    """

    def __init__(self, s=1.0):
        self.s = s
        self.scale = lambda l: [cv2.resize(frame, (int(84 / s), int(84 / s)), interpolation=cv2.INTER_AREA) for frame in
                                l]


class MaskedBaseWrapperDict(gym.ObservationWrapper):
    """
    Class for providing observation that contain multiple different representations of the same state. The observations are dictionaries and each value is a different input representation.
    """

    def __init__(self, env, buffer_window_size=4, extension=0, work_in_output_shape_planes=False,
                 work_in_output_shape_binary_mask=False, ):
        """
        Args:
            env (gym.Env, OCAtari): The environment to wrap (Should contain an OCAtari in the stack).
            buffer_window_size (int): How many observations to stack.
            extension (int): Specifies by how many pixels the bounding boxes are extended in each direction. The enlargement is only applied to values in the dictionary whose key ends with "extended"
            work_in_output_shape_planes (bool): Directly work in the 84x84 planes instead of downscaling the produced 210x160 ones.
            work_in_output_shape_binary_mask(bool): Directly work with the 84x84 binary masks instead of downscaling the produced 210x160 ones.
        """
        super().__init__(env)
        try:
            env.unwrapped.ale  # noqa: test for ale
            env.objects  # noqa: test for objects
        except AttributeError as e:
            raise AttributeError("Please use OCAtari with this wrapper.") from e

        self.work_in_output_shape_planes = work_in_output_shape_planes
        self.extension = extension
        self.object_types_v2 = {k: i for i, k in enumerate(
            get_max_objects(env.game_name, env.hud).keys())}
        self.object_types_v1 = {k: i for i, k in enumerate(get_class_dict(env.game_name).keys())}
        num_planes = len(self.object_types_v1)
        width_parallel_planes = len(self.object_types_v2) * 84
        length_planes = num_planes * buffer_window_size
        length_planes_combined = math.ceil(num_planes / 2) * buffer_window_size
        self.shape_binary_mask = (buffer_window_size, 84, 84)
        self.shape_planes = (length_planes, 84, 84)
        self.shape_planes_combined = (length_planes_combined, 84, 84)
        self.shape_parallel_planes = (self.buffer_window_size, 84, width_parallel_planes)
        self.observation_space = (
            gym.spaces.Dict({
                "binary_mask_extended": gym.spaces.Box(0, 255.0, self.shape_binary_mask),
                "binary_mask": gym.spaces.Box(0, 255.0, self.shape_binary_mask),
                "planes": gym.spaces.Box(0, 255.0, self.shape_planes),
                "planes_extended": gym.spaces.Box(0, 255.0, self.shape_planes),
                "parallel_planes": gym.spaces.Box(0, 255.0, self.shape_parallel_planes),
                "parallel_planes_extended": gym.spaces.Box(0, 255.0, self.shape_parallel_planes),
                "pixel_screen": gym.spaces.Box(0, 255.0, (self.buffer_window_size, 84, 84)),
                "pixel_screen_og": gym.spaces.Box(0, 255.0, (self.buffer_window_size, 210, 160)),
                "pixel_screen_color": gym.spaces.Box(0, 255.0, (3 * self.buffer_window_size, 210, 160)),
                "planes_combined": gym.spaces.Box(0, 255.0, self.shape_planes_combined),
                "planes_combined2": gym.spaces.Box(0, 255.0, self.shape_planes),
            }))
        self.buffer_window_size = buffer_window_size
        self._buffer_planes = deque([], maxlen=length_planes)
        self._buffer_planes_scaled_0point8 = deque([], maxlen=length_planes)
        self._buffer_planes_scaled_1point2 = deque([], maxlen=length_planes)
        self._buffer_planes_scaled_1point5 = deque([], maxlen=length_planes)
        self._buffer_binary_mask_scaled_0point8 = deque([], maxlen=buffer_window_size)
        self._buffer_binary_mask_scaled_1point2 = deque([], maxlen=buffer_window_size)
        self._buffer_binary_mask_scaled_1point5 = deque([], maxlen=buffer_window_size)
        self._buffer_binary_mask_scaled_kr = deque([], maxlen=buffer_window_size)
        self._buffer_planes_combined2 = deque([], maxlen=length_planes)
        self._buffer_planes_combined = deque([], maxlen=length_planes_combined)
        self._buffer_planes_extended = deque([], maxlen=length_planes)
        self._buffer_parallel_planes = deque([], maxlen=self.buffer_window_size)
        self._buffer_binary_mask = deque([], maxlen=buffer_window_size)
        self._buffer_object_mask = deque([], maxlen=buffer_window_size)
        self._buffer_parallel_planes_extended = deque([], maxlen=self.buffer_window_size)
        self._buffer_binary_mask_extended = deque([], maxlen=buffer_window_size)
        self._buffer_pixel_screen = deque([], maxlen=self.buffer_window_size)
        self._buffer_pixel_screen_og = deque([], maxlen=self.buffer_window_size)
        self._buffer_pixel_screen_color = deque([], maxlen=self.buffer_window_size)

        # for saving the current grayscale game screen and masked state
        self.pixel_screen = None
        self.state = None

        self.calc_limits_extended_wios = lambda x, y, x_w, y_h: (
            max(0, (y - self.extension) * 84 // 210),
            min((y_h + self.extension) * 84 // 210 + 1, 84),
            max(0, (x - self.extension) * 84 // 160),
            min((x_w + self.extension) * 84 // 160 + 1, 84)
        )
        self.calc_limits_wios = lambda x, y, x_w, y_h: (
            max(0, y * 84 // 210),
            min(y_h * 84 // 210 + 1, 84),
            max(0, x * 84 // 160),
            min(x_w * 84 // 160 + 1, 84)
        )
        self.maybe_scale_wios = lambda l: l  # no downscaling necessary

        self.calc_limits_extended = lambda x, y, x_w, y_h: (
            max(0, y - self.extension),
            min(y_h + self.extension, 210),
            max(0, x - self.extension),
            min(x_w + self.extension, 160)
        )
        self.calc_limits = lambda x, y, x_w, y_h: (
            max(0, y),
            min(y_h, 210),
            max(0, x),
            min(x_w, 160)
        )

        self.scale = lambda l: [cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in
                                l]  # downscale frames
        self.scale_0point8 = Scaler(0.8).scale
        self.scale_1point2 = Scaler(1.2).scale
        self.scale_1point5 = Scaler(1.5).scale
        self.scale_kr = lambda l: [cv2.resize(frame, (73, 96), interpolation=cv2.INTER_AREA) for frame in
                                   l]

        self.not_scale = lambda l: l

        self.maybe_scale_parallel = self.not_scale
        self.calc_limits_parallel = self.calc_limits_wios
        self.calc_limits_parallel_extended = self.calc_limits_extended_wios
        # bug
        # self.working_shape_parallel = (buffer_window_size, 84, width_parallel_planes)
        # bug fix
        self.working_shape_parallel = (1, 84, width_parallel_planes)

        if work_in_output_shape_planes:
            self.maybe_scale_planes = self.not_scale
            self.calc_limits_planes = self.calc_limits_wios
            self.calc_limits_planes_extended = self.calc_limits_extended_wios
            self.working_shape_planes = (num_planes, 84, 84)
            self.working_shape_planes_combined = (math.ceil(num_planes / 2), 84, 84)
            self.working_shape_planes_combined2 = (num_planes, 84, 84)
        else:
            self.maybe_scale_planes = self.scale
            self.calc_limits_planes = self.calc_limits
            self.calc_limits_planes_extended = self.calc_limits_extended
            self.working_shape_planes = (num_planes, 210, 160)
            self.working_shape_planes_combined = (math.ceil(num_planes / 2), 210, 160)
            self.working_shape_planes_combined2 = (num_planes, 210, 160)

        if work_in_output_shape_binary_mask:
            self.maybe_scale_binary_mask = self.not_scale
            self.calc_limits_binary_mask = self.calc_limits_wios
            self.calc_limits_binary_mask_extended = self.calc_limits_extended_wios
            self.working_shape_binary_mask = (1, 84, 84)
        else:
            self.maybe_scale_binary_mask = self.scale
            self.calc_limits_binary_mask = self.calc_limits
            self.calc_limits_binary_mask_extended = self.calc_limits_extended
            self.working_shape_binary_mask = (1, 210, 160)

        self.current_pixel_screen = lambda: self.unwrapped.ale.getScreenGrayscale()  # noqa: OCAtari in the env stack

    def observation(self, observation):
        self.pixel_screen = self.current_pixel_screen()  # noqa: only used when somethis is returned
        # print(self.pixel_screen.shape)
        pixel_screen = self.pixel_screen
        self.pixel_screen_color = self.unwrapped.ale.getScreenRGB()
        scaled_pixel_screen = cv2.resize(
            self.pixel_screen,
            (84, 84),
            interpolation=cv2.INTER_AREA
        ).astype(np.uint8)

        self.state = {
            "binary_mask_extended": np.zeros(self.working_shape_binary_mask, dtype=np.uint8),
            "binary_mask": np.zeros(self.working_shape_binary_mask, dtype=np.uint8),
            "object_mask": np.zeros(self.working_shape_binary_mask, dtype=np.uint8),
            "planes_extended": np.zeros(self.working_shape_planes, dtype=np.uint8),
            "planes": np.zeros(self.working_shape_planes, dtype=np.uint8),
            "parallel_planes": np.zeros(self.working_shape_parallel, dtype=np.uint8),
            "parallel_planes_extended": np.zeros(self.working_shape_parallel, dtype=np.uint8),
            "planes_combined": np.zeros(self.working_shape_planes_combined, dtype=np.uint8),
            "planes_combined2": np.zeros(self.working_shape_planes_combined2, dtype=np.uint8),
            "pixel_screen": scaled_pixel_screen,
            "pixel_screen_og": pixel_screen,
            "obj": observation,

        }

        for o in self.env.objects:  # noqa: OCAtari in the stack
            if not (o is None or o.category == "NoObject"):
                x, y, w, h = o.xywh
                x_w = x + w
                y_h = y + h
                if x_w > 0 and y_h > 0:
                    self.set_value_planes(*self.calc_limits_planes(x, y, x_w, y_h), o, "planes")
                    self.set_value_planes(*self.calc_limits_planes_extended(x, y, x_w, y_h), o, "planes_extended")
                    self.set_value_binary_mask(*self.calc_limits_binary_mask(x, y, x_w, y_h), o, "binary_mask")
                    self.set_value_binary_mask(*self.calc_limits_binary_mask_extended(x, y, x_w, y_h), o,
                                               "binary_mask_extended")
                    self.set_value_object_mask(*self.calc_limits_binary_mask(x, y, x_w, y_h), o, "object_mask")
                    self.set_value_parallel_planes(*self.calc_limits_parallel(x, y, x_w, y_h), o, "parallel_planes")
                    self.set_value_parallel_planes(*self.calc_limits_parallel_extended(x, y, x_w, y_h), o,
                                                   "parallel_planes_extended")
                    self.set_value_combined(*self.calc_limits_planes(x, y, x_w, y_h), o, "planes_combined")
                    self.set_value_combined2(*self.calc_limits_planes(x, y, x_w, y_h), o, "planes_combined2")

        return self.create_obs(self.state)

    def show_img(sef, img, title):
        plt.imshow(img, cmap='gray')
        plt.axis('off')  # Hide axes
        plt.title(title)
        plt.show()

    def create_obs(self, obs):
        self._buffer_pixel_screen_color.extend([self.pixel_screen_color])
        self._buffer_pixel_screen_og.extend([obs["pixel_screen_og"]])
        self._buffer_planes.extend(self.maybe_scale_planes(obs["planes"]))
        self._buffer_planes_extended.extend(self.maybe_scale_planes(obs["planes_extended"]))
        self._buffer_binary_mask.extend(self.maybe_scale_binary_mask(obs["binary_mask"]))
        self._buffer_object_mask.extend(self.maybe_scale_binary_mask(obs["object_mask"]))
        self._buffer_parallel_planes.extend(self.maybe_scale_parallel(obs["parallel_planes"]))
        self._buffer_parallel_planes_extended.extend(self.maybe_scale_parallel(obs["parallel_planes_extended"]))
        self._buffer_binary_mask_extended.extend(self.maybe_scale_binary_mask(obs["binary_mask_extended"]))
        self._buffer_pixel_screen.extend([obs["pixel_screen"]])
        self._buffer_planes_combined.extend(self.maybe_scale_planes(obs["planes_combined"]))
        self._buffer_planes_combined2.extend(self.maybe_scale_planes(obs["planes_combined2"]))
        if not self.work_in_output_shape_planes:
            self._buffer_planes_scaled_0point8.extend(self.scale_0point8(obs["planes"]))
            self._buffer_planes_scaled_1point2.extend(self.scale_1point2(obs["planes"]))
            self._buffer_planes_scaled_1point5.extend(self.scale_1point5(obs["planes"]))
            self._buffer_binary_mask_scaled_0point8.extend(self.scale_0point8(obs["binary_mask"]))
            self._buffer_binary_mask_scaled_1point2.extend(self.scale_1point2(obs["binary_mask"]))
            self._buffer_binary_mask_scaled_1point5.extend(self.scale_1point5(obs["binary_mask"]))
            self._buffer_binary_mask_scaled_kr.extend(self.scale_kr(obs["binary_mask"]))
        '''
        for z in np.asarray(self._buffer_binary_mask):
            self.show_img(z[0], "xd")
        '''
        created_obs = {
            "binary_mask_extended": np.asarray(self._buffer_binary_mask_extended),
            "binary_mask": np.asarray(self._buffer_binary_mask),
            "object_mask": np.asarray(self._buffer_object_mask),
            "parallel_planes": np.asarray(self._buffer_parallel_planes),
            "parallel_planes_extended": np.asarray(self._buffer_parallel_planes_extended),
            "planes": np.asarray(self._buffer_planes),
            "planes_extended": np.asarray(self._buffer_planes_extended),
            "pixel_screen": np.asarray(self._buffer_pixel_screen),
            "planes_combined": np.asarray(self._buffer_planes_combined),
            "planes_combined2": np.asarray(self._buffer_planes_combined2),
            "pixel_screen_og": np.asarray(self._buffer_pixel_screen_og),
            "pixel_screen_color": np.asarray(self._buffer_pixel_screen_color),
            "obj": obs["obj"],
        }
        if not self.work_in_output_shape_planes:
            created_obs["planes_0.8"] = np.asarray(self._buffer_planes_scaled_0point8)
            created_obs["planes_1.2"] = np.asarray(self._buffer_planes_scaled_1point2)
            created_obs["planes_1.5"] = np.asarray(self._buffer_planes_scaled_1point5)
            created_obs["binary_mask_0.8"] = np.asarray(self._buffer_binary_mask_scaled_0point8)
            created_obs["binary_mask_1.2"] = np.asarray(self._buffer_binary_mask_scaled_1point2)
            created_obs["binary_mask_1.5"] = np.asarray(self._buffer_binary_mask_scaled_1point5)
            created_obs["binary_mask_kr"] = np.asarray(self._buffer_binary_mask_scaled_kr)
        # print(created_obs["pixel_screen_og"].shape)
        return created_obs

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)

        # fill buffer
        for _ in range(self.buffer_window_size):
            obs = self.observation(ret[0])

        return obs, *ret[1:]  # noqa: cannot be undefined

    def set_value_binary_mask(self, y_min, y_max, x_min, x_max, o, key):
        self.state[key][0, y_min:y_max, x_min:x_max].fill(255)

    def set_value_parallel_planes(self, y_min, y_max, x_min, x_max, o, key):
        offset = self.object_types_v2[o.category] * 84
        self.state[key][0, y_min:y_max, offset + x_min: offset + x_max].fill(255)

    def set_value_planes(self, y_min, y_max, x_min, x_max, o, key):
        self.state[key][self.object_types_v1[o.category], y_min:y_max, x_min:x_max].fill(255)

    def set_value_combined2(self, y_min, y_max, x_min, x_max, o, key):
        if o.category == "Player":
            for j in range(len(self.object_types_v1)):
                self.state[key][j, y_min:y_max, x_min:x_max].fill(255)
        else:
            self.state[key][self.object_types_v1[o.category], y_min:y_max, x_min:x_max].fill(255)

    def set_value_combined(self, y_min, y_max, x_min, x_max, o, key):
        self.state[key][int(self.object_types_v1[o.category] / 2), y_min:y_max, x_min:x_max].fill(255)

    def set_value_object_mask(self, y_min, y_max, x_min, x_max, o, key):
        self.state[key][0, y_min:y_max, x_min:x_max] = self.pixel_screen[y_min:y_max, x_min:x_max]
