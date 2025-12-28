import math
import numpy as np
import gymnasium as gym
from collections import deque
from ocatari.ram.extract_ram_info import get_class_dict, get_max_objects, get_object_state_size


def is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2, offset):
    left1, right1 = x1 - offset, x1 + w1 + offset
    top1, bottom1 = y1 - offset, y1 + h1 + offset

    left2, right2 = x2 - offset, x2 + w2 + offset
    top2, bottom2 = y2 - offset, y2 + h2 + offset

    return not (right1 <= left2 or right2 <= left1 or
                bottom1 <= top2 or bottom2 <= top1)


class ObjExtended(gym.ObservationWrapper):
    """
    Class for the extended Obj input representation/obs_mode.
    """

    def __init__(self, env, use_distances=True, use_direction=True,
                 multiply_player_info=True,
                 use_angle=True, use_overlap=True, use_object_xy=True, double_input=True,
                 apply_centerpoints=False, use_origin_angle=True, use_origin_distances=True, use_vel=True,
                 use_time_angle=True, use_time_distances=True, overlap_offset=2, normalize=True, ):
        """
        Args:
            env (gym.Env, OCAtari): The environment to wrap. (OCAtari needs to be in the stack)
            use_distances (bool): If True, add the distance feature to the observation.
            use_direction (bool): If True, add the direction feature to the observation.
            multiply_player_info (bool): If True, add the multiply_player_info feature to the observation.
            use_angle (bool): If True, add the angle feature to the observation.
            use_overlap (bool): If True, add the overlap feature to the observation.
            use_object_xy (bool): If True, the observation includes each object's xy coordinates.
            double_input (bool): If True, the input x is doubled by concatenating it with itself: [x,x].
            apply_centerpoints (bool): If True, centerpoints are used to compute the features
            use_origin_angle (bool): If True, the observation includes the origin angle feature
            use_origin_distances (bool): If True, the observation includes the origin distance feature
            use_vel (bool): If True, add the velocity feature to the observation.
            use_time_angle (bool): If True, add the time angle feature to the observation.
            use_time_distances (bool): If True, add the time distance feature to the observation.
            overlap_offset (int): The bounding boxes used to compute the overlap feature are increased in each direction by this number of pixels
            normalize (bool): If True, normalize the features distances, angles and overlaps before adding to the observation.
        """
        super().__init__(env)
        try:
            env.unwrapped.ale  # noqa: test for ale
            env.objects  # noqa: test for objects
        except AttributeError as e:
            raise AttributeError("Please use OCAtari with this wrapper.") from e
        self.env = env
        self.normalize = normalize
        self.max_distance = math.ceil(math.sqrt(210 ** 2 + 160 ** 2))
        self.max_value = 255  # the maximum value any normalized scalar feature can have, another value can be chosen but it should be close to 210
        self.use_distances = use_distances
        self.use_direction = use_direction
        self.use_angle = use_angle
        self.use_overlap = use_overlap
        self.multiply_player_info = multiply_player_info
        self.use_object_xy = use_object_xy
        self.double_input = double_input
        self.use_origin_angle = use_origin_angle
        self.use_origin_distances = use_origin_distances
        self.use_vel = use_vel
        self.use_time_angle = use_time_angle
        self.use_time_distances = use_time_distances
        self.overlap_offset = overlap_offset
        self.apply_centerpoints = apply_centerpoints

        self.game_name = env.game_name
        self.hud = env.hud
        self.buffer_window_size = env.buffer_window_size
        self.max_objects_per_cat = get_max_objects(
            self.game_name, self.hud)
        # Create a dictionary of game object classes for categorization
        self._class_dict = get_class_dict(self.game_name)
        # Initialize slots to store all possible game objects
        self._slots = [self._class_dict[c]()
                       for c, n in self.max_objects_per_cat.items()
                       for _ in range(n)]

        self.indices_of_objs_xy_info = []
        self.indices_of_objs_width_info = []

        for idx, o in enumerate(self._slots):
            if o._ns_meaning[0] == "WIDTH":
                self.indices_of_objs_width_info.append(idx)
            elif o._ns_meaning[0] == "POSITION":
                self.indices_of_objs_xy_info.append(idx)
            elif o._ns_meaning[0] == "x" and o._ns_meaning[1] == "y":
                self.indices_of_objs_xy_info.append(idx)
            else:
                raise NotImplementedError()

        self.number_objects_with_xy = len(self.indices_of_objs_xy_info)

        number_of_scalar_features = 0
        number_of_scalar_features += get_object_state_size(env.game_name, env.hud) if use_object_xy else len(
            self.indices_of_objs_width_info)
        number_of_scalar_features += self.number_objects_with_xy if use_distances else 0
        number_of_scalar_features += self.number_objects_with_xy * 2 if use_direction else 0
        number_of_scalar_features += self.number_objects_with_xy * 2 if multiply_player_info else 0
        number_of_scalar_features += self.number_objects_with_xy if use_angle else 0
        number_of_scalar_features += self.number_objects_with_xy if use_overlap else 0
        number_of_scalar_features += self.number_objects_with_xy if use_origin_angle else 0
        number_of_scalar_features += self.number_objects_with_xy if use_origin_distances else 0
        number_of_scalar_features += self.number_objects_with_xy * 2 if use_vel else 0
        number_of_scalar_features += self.number_objects_with_xy if use_time_angle else 0
        number_of_scalar_features += self.number_objects_with_xy if use_time_distances else 0
        self.working_shape = (self.buffer_window_size,
                              number_of_scalar_features if not double_input
                              else number_of_scalar_features * 2)
        self.observation_space = (
            gym.spaces.Box(
                -self.max_value, self.max_value,
                self.working_shape, ))

        self._buffer = deque([], maxlen=self.buffer_window_size)

    def extend_obs(self, observation):
        objects = [self.env.objects[idx] for idx in self.indices_of_objs_xy_info]
        widths = [self.env.objects[idx]._nsrepr[0] for idx in self.indices_of_objs_width_info]

        # Extract xy and wh in one go
        xy = np.array([o._nsrepr[0:2] for o in objects], )  # dtype=float)
        wh = np.array([o.wh for o in objects], )  # dtype=float)

        # Centerpoints (avoids recomputation later)
        centerpoints = xy + wh / 2.0

        # Start feature collection
        features = []

        # Player info replicated
        if self.multiply_player_info:
            obs_slice = xy[0]
            stacked = np.tile(obs_slice, (len(objects), 1))
            features.append(stacked)

        # XY positions
        if self.use_object_xy:
            features.append(xy)

        # Distances
        if self.use_distances:
            base = centerpoints if self.apply_centerpoints else xy
            distances = np.linalg.norm(base - base[0], axis=1, keepdims=True)
            distances = self.max_value * distances / self.max_distance if self.normalize else distances
            features.append(distances)

        # Directions
        if self.use_direction:
            base = centerpoints if self.apply_centerpoints else xy
            directions = base - base[0]
            features.append(directions)

        # Angles
        if self.use_angle:
            base = centerpoints if self.apply_centerpoints else xy
            directions = base - base[0]
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            angles = ((angles + math.pi) * self.max_value/ (2 * math.pi)) if self.normalize else angles
            angles = angles[:, np.newaxis]
            features.append(angles)

        # Overlaps
        if self.use_overlap:
            x2, y2 = xy[0]
            w2, h2 = objects[0].wh
            overlaps = []
            for (x1, y1), (w1, h1) in zip(xy, wh):
                if (w1 == 0 and h1 == 0) or (w2 == 0 and h2 == 0):
                    overlaps.append(0)
                else:
                    if self.normalize:
                        overlaps.append(self.max_value if is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2, self.overlap_offset) else 0)
                    else:
                        overlaps.append(1 if is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2,self.overlap_offset) else 0)
            features.append(np.array(overlaps)[:, np.newaxis])

        # Distances
        if self.use_origin_distances:
            base = centerpoints if self.apply_centerpoints else xy
            distances = np.linalg.norm(base, axis=1, keepdims=True)
            distances = self.max_value * distances / self.max_distance if self.normalize else distances
            features.append(distances)

        if self.use_origin_angle:
            base = centerpoints if self.apply_centerpoints else xy
            angles = np.arctan2(base[:, 1], base[:, 0])
            angles = ((angles + math.pi) * self.max_value/ (2 * math.pi)) if self.normalize else angles
            angles = angles[:, np.newaxis]
            features.append(angles)

        vels = np.array(
            [[self.env.objects[idx].dx, self.env.objects[idx].dy] for idx in self.indices_of_objs_xy_info])
        if self.use_vel:
            features.append(vels)

        # Distances
        if self.use_time_distances:
            distances = np.linalg.norm(vels, axis=1, keepdims=True)
            distances = self.max_value * distances / self.max_distance if self.normalize else distances
            features.append(distances)

        # Angles
        if self.use_time_angle:
            angles = np.arctan2(vels[:, 1], vels[:, 0])
            angles = ((angles + math.pi) * self.max_value / (2 * math.pi)) if self.normalize else angles
            angles = angles[:, np.newaxis]
            features.append(angles)

        # Concatenate all features
        new_obs = np.concatenate(features, axis=-1).flatten()

        # Add width info
        if widths:
            new_obs = np.concatenate((np.array(widths), new_obs))

        # Duplicate if required
        if self.double_input:
            new_obs = np.concatenate((new_obs, new_obs))

        return new_obs

    def observation(self, observation):
        new_observation2 = self.extend_obs(observation)
        self._buffer.append(new_observation2)
        r_obs = np.array(self._buffer)
        return r_obs

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        for _ in range(self.buffer_window_size):
            self._buffer.append(ret[0][0])
        obs = np.array(self._buffer)
        return obs, *ret[1:]  # noqa: cannot be undefined
