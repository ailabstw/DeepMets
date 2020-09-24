import torch
import torch.nn.functional as F

import numpy as np


class Normalize(object):
    """
    1. If mean / variance are not set, it will
        * Automatically calculate from the dataset

    2. If mean / variance is set to a number
        * the data will be normalized using these mean and variance

    3. If mean / variance is set to a list of numbers
        * the number of elements in this list should correspond to channel dimension of the data
        * the data will be normalized along its channel dimension
    """

    def __init__(
            self,
            mean=None,
            std=None):

        self.mean = mean
        self.std = std

    def calculate_mean_variance(self, data_dict):
        """
        Calculate mean / variance
        Args:
            data_dict: [dict], it should contain a key, called "data"
        Returns:
            mean: [float], the mean of data_dict["data"]
            std: [float], the standard deviation of data_dict["data"]
        """

        target_axes = tuple([i + 1 for i in range(len(data_dict["data"].shape) - 1)])
        mean = np.mean(data_dict["data"], axis=target_axes)
        std = np.std(data_dict["data"], axis=target_axes)

        return mean, std

    def __call__(self, data_dict):
        """
        Args:
            data_dict: [dict], it should contain a key, called "data"
        Returns:
            data_dict: [dict] with data_dict["data"] being normalized
        """

        gotTensor = False
        if isinstance(data_dict["data"], torch.Tensor):

            gotTensor = True
            data_dict["data"] = data_dict["data"].numpy()

        mean, std = self.mean, self.std

        if mean is None and std is None:
            mean, std = self.calculate_mean_variance(data_dict)

        if isinstance(mean, (list, tuple, np.ndarray)) and isinstance(std, (list, tuple, np.ndarray)):
            if len(mean) != len(std):
                raise ValueError("The number of mean values is not the same with that of std values! \
                                  Please check!")

        if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):

            if isinstance(mean, (list, tuple, np.ndarray)) and len(mean) == data_dict["data"].shape[0]:

                mean = np.array(mean)[:, None, None, None] if len(data_dict["data"].shape) - 1 == 3 \
                    else np.array(mean)[:, None, None]
                std = np.array(std)[:, None, None, None] if len(data_dict["data"].shape) - 1 == 3 \
                    else np.array(std)[:, None, None]

            elif isinstance(mean, (list, tuple, np.ndarray)) and len(mean) == data_dict["data"].shape[1]:

                mean = np.repeat(np.expand_dims(np.array(mean)[:, None, None, None], axis=0),
                                 data_dict["data"].shape[0],
                                 axis=0)
                std = np.repeat(np.expand_dims(np.array(std)[:, None, None, None], axis=0),
                                data_dict["data"].shape[0],
                                axis=0)
            else:

                raise ValueError("The number of means values - {} - should correspond to \
                the number of channels of the volume (or split volume- {})!".format(len(mean), data_dict["data"].shape))

        data_dict["data"] = (data_dict["data"] - mean) / std

        if gotTensor:
            data_dict["data"] = torch.FloatTensor(data_dict["data"])

        return data_dict


class Split(object):
    """
    Split the volume into pieces for heavy evaluation
    This class is applied on the data (in data_dict) retrieved by the target_key you provided

    Args:
        target_key: [str], the key to retrieve data in data_dict for Split
        size: [int] or [tuple], the (window) size of each split from a given 3D volume (or 2D image)
        margin: [int] or [tuple], the padding size for each cube
        feature_stride: [int], the overall stride from input to the last feature_map
        constant_value: [float], the constant value for padding
    """

    def __init__(self,
                 size,
                 margin,
                 feature_stride=1,
                 target_key=None,
                 constant_value=0.):

        if isinstance(size, int) and isinstance(margin, int):
            if isinstance(feature_stride, int):
                assert (size % feature_stride == 0), \
                    "Size should be divided by feature_stride (the overall stride of the network)!"
                assert (margin % feature_stride == 0), \
                    "Margin should be divided by feature_stride (the overall stride of the network)!"

            elif isinstance(feature_stride, (list, tuple, np.ndarray)):
                assert np.array([(size % i) == 0 for i in feature_stride]).all(), \
                    "Size should be divided by feature_stride (the overall stride of the network)!"
                assert np.array([(margin % i) == 0 for i in feature_stride]).all(), \
                    "Margin should be divided by feature_stride (the overall stride of the network)!"

        elif isinstance(size, (list, tuple, np.ndarray)) \
                and isinstance(margin, (list, tuple, np.ndarray)) \
                and None not in size:

            if isinstance(feature_stride, int):
                assert np.array([(i % feature_stride) == 0 for i in size]).all(), \
                    "Size should be divided by feature_stride (the overall stride of the network)!"
                assert np.array([(i % feature_stride) == 0 for i in margin]).all(), \
                    "Margin should be divided by feature_stride (the overall stride of the network)!"

            elif isinstance(feature_stride, (list, tuple, np.ndarray)):
                assert np.array([(i % j) == 0 for i, j in zip(size, feature_stride)]).all(), \
                    "Size should be divided by feature_stride (the overall stride of the network)!"
                assert np.array([(i % j) == 0 for i, j in zip(margin, feature_stride)]).all(), \
                    "Margin should be divided by feature_stride (the overall stride of the network)!"

        self.target_key = target_key
        if self.target_key is None:
            raise ValueError("target key - {} - for Split should be provided. \
            ( e.g. target_keys='data', in yaml => {target_keys: 'data'} )")

        self.size = size

        self.template_size = size
        self.determine_size_by_run = False

        if isinstance(self.size, (list, tuple, np.ndarray)) and None in self.size:
            self.determine_size_by_run = True
        self.margin = margin

        self.feature_stride = feature_stride
        self.constant_value = constant_value

        self.nzhw = None
        self.zhw = None

        self.n_splits = None

    def split_data(self, data):
        """
        Args:
            data: [np.ndarray], shape (channel, z, h, w), the channel dim will be ignored
        Returns:
            splits: [np.ndarray], shape (num_splits, channel, z, h, w)
        """

        zhw = data.shape[1:]

        nzhw = [int(np.ceil(float(i_shape) / self.size[idx])) for idx, i_shape in enumerate(zhw)]

        self.nzhw = nzhw
        self.zhw = zhw
        self.ndim = len(self.nzhw)

        pad = [[0, 0]] + [[self.margin[idx], int(i_n * self.size[idx] - zhw[idx] + self.margin[idx])]
                          for idx, i_n in enumerate(self.nzhw)]
        data = np.pad(data, pad, 'constant', constant_values=self.constant_value)

        splits = []

        for iz in range(self.nzhw[0]):
            for ih in range(self.nzhw[1]):
                for iw in range(self.nzhw[2]):
                    sz = iz * self.size[0]
                    ez = (iz + 1) * self.size[0] + 2 * self.margin[0]
                    sh = ih * self.size[1]
                    eh = (ih + 1) * self.size[1] + 2 * self.margin[1]
                    sw = iw * self.size[2]
                    ew = (iw + 1) * self.size[2] + 2 * self.margin[2]

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)

        return splits

    def __call__(self, data_dict):

        if self.target_key not in data_dict:
            raise KeyError("No key - {} - found in data_dict!".format(self.target_key))

        tensor_type = None
        if torch.is_tensor(data_dict[self.target_key]):
            tensor_type = data_dict[self.target_key].type()
            data_dict[self.target_key] = data_dict[self.target_key].numpy()

        if isinstance(self.size, int):
            self.size = [self.size] * len(data_dict[self.target_key].shape[1:])
        if isinstance(self.margin, int):
            self.margin = [self.margin] * len(data_dict[self.target_key].shape[1:])

        if self.determine_size_by_run:
            target_size = []
            for idx in range(len(self.size)):
                if self.template_size[idx] is not None:
                    target_size.append(self.template_size[idx])
                else:
                    target_size.append(data_dict[self.target_key].shape[idx + 1])
            self.size = target_size

        splits = self.split_data(data_dict[self.target_key])
        self.n_splits = len(splits)

        data_dict[self.target_key] = splits

        if tensor_type is not None:
            data_dict[self.target_key] = torch.from_numpy(data_dict[self.target_key]).type(tensor_type)

        data_dict[self.target_key] = splits
        data_dict["split_info"] = {"size": self.size,
                                   "margin": self.margin,
                                   "feature_stride": self.feature_stride,
                                   "nzhw": self.nzhw,
                                   "zhw": self.zhw,
                                   "target_key": self.target_key,
                                   "n_splits": self.n_splits}

        return data_dict


class ToTensor(object):
    """
    Convert np.array to torch tensors

    Args:
        key_type_pairs: None or [dict] -> <key: type>
            1. the key should be also in data_dict
            2. the type should be one of the torch tensor types
    """

    def __init__(self, key_type_pairs=None):

        super().__init__()

        if not isinstance(key_type_pairs, dict):
            raise ValueError("key_type_pairs should at least be: {data: torch.FloatTensor}!")

        self.key_type_pairs = key_type_pairs

    def convert_to_float_tensor(self, data):
        """
        Convert to float tensor
        1. convert to np.float32 at first (avoid conversion error when the value is np.uint16)
        2. convert to torch.FloatTensor
        Returns:
            float tensor
        """

        data = data.astype(np.float32)

        return torch.from_numpy(data).type(torch.FloatTensor)

    def __call__(self, data_dict):

        for key, type in self.key_type_pairs.items():

            totype = type

            if key in data_dict:

                if totype == "torch.FloatTensor":

                    data_dict[key] = self.convert_to_float_tensor(data_dict[key])

                else:

                    data_dict[key] = torch.from_numpy(data_dict[key])

                    if totype is not None:

                        data_dict[key] = data_dict[key].type(eval(r"{}".format(totype)))

            else:

                raise ValueError("In ToTensor - key: {} not found in data_dict!".format(key))

        return data_dict


class Resize(object):
    """
    This is used to resize pytorch tensor

    Args:
        target_keys: [list], the keys to retrieve tensor in data_dict
        to_size: [tuple], the target size
        mode: [str], the mode for resize ("nearest", "bilinear", "trilinear" , ... )
        align_corners: [bool]
    """

    def __init__(self, target_keys, to_size, mode="nearest", align_corners=True):

        if not isinstance(target_keys, list) and isinstance(target_keys, str):
            target_keys = [target_keys]

        self.target_keys = target_keys
        self.to_size = to_size
        self.mode = mode
        self.align_corners = align_corners

    def interpolate(self, target):
        """
        Args:
            target: [torch.tensor], the tensor to be interpolated
        Returns:
            interpolated_target: [torch.tensor]
        """

        if not isinstance(self.to_size, (list, tuple, np.ndarray)):
            raise ValueError("To interpolate, please specify to_size for each dimension of target data \
                              (e.g. (128, 128, 128) or (128, 128)!)")

        if len(target.shape) == len(self.to_size) + 1:

            if "linear" in self.mode:

                return torch.squeeze(F.interpolate(torch.unsqueeze(target, dim=0),
                                                   self.to_size,
                                                   mode=self.mode,
                                                   align_corners=self.align_corners),
                                     dim=0)

            return torch.squeeze(F.interpolate(torch.unsqueeze(target,
                                                               dim=0),
                                               self.to_size,
                                               mode=self.mode),
                                 dim=0)

        elif len(target.shape) == len(self.to_size) + 2:

            if "linear" in self.mode:
                return F.interpolate(target,
                                     self.to_size,
                                     mode=self.mode,
                                     align_corners=self.align_corners)

            return F.interpolate(target, self.to_size, mode=self.mode)

        elif len(target.shape) == len(self.to_size) + 3:

            if "linear" in self.mode:
                return torch.cat([torch.unsqueeze(F.interpolate(i,
                                                                self.to_size,
                                                                mode=self.mode,
                                                                align_corners=self.align_corners),
                                                  dim=0)
                                  for i in target],
                                 dim=0)

            return torch.cat([torch.unsqueeze(F.interpolate(i,
                                                            self.to_size,
                                                            mode=self.mode),
                                              dim=0)
                              for i in target],
                             dim=0)
        else:
            raise ValueError("Unknown target size {} for {} interpolation! Please check!".format(target.shape,
                                                                                                 len(self.to_size)))

    def convert_to_torch_tensor(self, array):
        '''
        This is used to convert input array to torch tensor if it is not a torch tensor
        Convert array to torch.FloatTensor for interpolation

        Args:
            array: either [np.ndarray] or [torch.Tensor]

        Returns:
            array: [torch.Tensor]
            convert_from_type: [str] or None
                If array is a torch tensor, the returned convert_from_type is None
        '''

        if isinstance(array, (np.ndarray)):

            source_type = array.dtype
            array = array.astype(np.float64)

            return torch.from_numpy(array).type(torch.FloatTensor), source_type

        else:

            return array, None

    def convert_to_np(self, array, to_numpy_type):
        '''
        This is used to convert torch tensor back to its original np array type
        Args:
            array: [torch.Tensor]
            to_numpy_type: [str] or None
        Returns
            array: [np.ndarray]
        '''
        return array.numpy().astype(to_numpy_type)

    def __call__(self, data_dict):

        for target_key in self.target_keys:

            if target_key in data_dict:

                target = data_dict[target_key]

                target, convert_from_type = self.convert_to_torch_tensor(target)

                data_dict[target_key] = self.interpolate(target)

                if convert_from_type is not None:
                    data_dict[target_key] = self.convert_to_np(data_dict[target_key], convert_from_type)

            else:
                raise KeyError("Not found key - {} - in data_dict!".format(target_key))

        return data_dict
