import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(NormConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        return F.normalize(out, p=2, dim=1)


"""Utility functions for visualizing embeddings."""


def get_principle_components(embedding, num_components=3):
    """Calculates the principal components given the embedding features.

    Args:
        embedding: A 2-D float tensor with shape `[batch x h x w, embedding_dim]`.
        num_components: The number of principal components to return.

    Returns:
        A 2-D float tensor with principal components in the last dimension.
    """
    embedding -= torch.mean(embedding, dim=0, keepdim=True)
    sigma = torch.matmul(embedding.transpose(0, 1), embedding)
    u, _, _ = torch.svd(sigma)
    return u[:, :num_components]


def pca(embedding, num_components=3):
    """Conducts principal component analysis on the embedding features.

    This function is used to reduce the dimensionality of the embedding, so that
    we can visualize the embedding as an RGB image.

    Args:
        embedding: A 4-D float tensor with shape `[batch, embedding_dim, h, w]`.
        num_components: The number of principal components to be reduced to.

    Returns:
        A 4-D float tensor with shape [batch, num_components, height, width].
    """
    N, c, h, w = embedding.size()
    embedding = embedding.permute([0, 2, 3, 1]).reshape([-1, c])

    pc = get_principle_components(embedding, num_components)
    embedding = torch.matmul(embedding, pc)
    embedding = embedding.reshape([N, h, w, -1]).permute([0, 3, 1, 2])
    return embedding


""" Generate location features"""


def generate_location_features(batch_size, img_dimensions, cuda=True):
    """Calculates location features for an image.

    This function generates location features for an image. The 2-D location
    features range from 0 to 1 for y and x axes each.

    Args:
        img_dimensions: A list of 2 integers for image's y and x dimension.
        cuda: Whether to use cuda backend.

    Returns:
        A 4-D float32 tensor with shape `[batch, img_dimension[0], img_dimension[1], 2]`.
    """
    y_features = np.linspace(0, 1, img_dimensions[0])
    x_features = np.linspace(0, 1, img_dimensions[1])

    x_features, y_features = np.meshgrid(x_features, y_features)
    location_features = np.expand_dims(np.stack([y_features, x_features], axis=2), 0)
    location_features = torch.from_numpy(location_features).float()
    if cuda:
        location_features = location_features.cuda()
    return torch.cat([location_features for _ in range(batch_size)], dim=0)


"""Decode segmentation map given one-channel predicted tensor."""


def decode_seg_map(label_mask, num_classes):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): a (M,N) array of integer values denoting
            the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
            in a figure.
        num_classes: the number of classes to decode.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    label_colours = np.asarray([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
        [192, 0, 192],
        [30, 30, 30],
        [158, 30, 30],
        [30, 158, 30],
        [158, 158, 30],
        [30, 30, 158],
        [158, 30, 158],
        [30, 158, 158],
        [158, 158, 158],
        [94, 30, 30],
        [222, 30, 30],
        [94, 158, 30],
        [222, 158, 30],
        [94, 30, 158],
        [222, 30, 158],
        [94, 158, 158],
        [222, 158, 158],
        [30, 94, 30],
        [158, 94, 30],
        [30, 222, 30],
        [158, 222, 30],
        [30, 94, 158],
        [222, 30, 222],
        [60, 60, 60],
        [188, 60, 60],
        [60, 188, 60],
        [188, 188, 60],
        [60, 60, 188],
        [188, 60, 188],
        [60, 188, 188],
        [188, 188, 188],
        [124, 60, 60],
        [252, 60, 60],
        [124, 188, 60],
        [252, 188, 60],
        [124, 60, 188],
        [252, 60, 188],
        [124, 188, 188],
        [252, 188, 188],
        [60, 124, 60],
        [188, 124, 60],
        [60, 252, 60],
        [188, 252, 60],
        [60, 124, 188],
        [252, 60, 252],
        [90, 90, 90],
        [218, 90, 90],
        [90, 218, 90],
        [218, 218, 90],
        [90, 90, 218],
        [218, 90, 218],
        [90, 218, 218],
        [218, 218, 218],
        [154, 90, 90],
        [255, 90, 90],
        [154, 218, 90],
        [255, 218, 90],
        [154, 90, 218],
        [255, 90, 218],
        [154, 218, 218],
        [255, 218, 218],
        [90, 154, 90],
        [218, 154, 90],
        [90, 255, 90],
        [218, 255, 90],
        [90, 154, 218],
        [255, 90, 255]
    ])
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, num_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


"""Replace element(s) in an array according to a dict."""


def replace_array_ele_as_dict(array, d, replace_with=None):
    """
    A very slow implementation that replace element(s) in an array according
    to a dict.

    Args:
        array: array to be replace.
        d: replace dictionary {old_ele: new_ele}.
        replace_with: replace the ele not in d.keys() into replace_with, when None,
            leave it as what it was.

    Returns:
        array: replaced numpy array.
    """
    old_shape = array.shape
    old_type = array.dtype
    array = array.flatten().tolist()
    if replace_with is not None:
        array = [d[idx] if idx in d else replace_with for idx in array]
    else:
        array = [d[idx] if idx in d else idx for idx in array]
    array = np.array(array, dtype=old_type).reshape(old_shape)
    return array


"""Sort dict by key or value and return a list of (k, v)."""


def sort_dict_by(x, by='key', reverse=False):
    """
    Sort dict by key or value and return a list of (k, v).

    Args:
        x: a dict {k:v}.
        by: `key` or 'value'.
        reverse: whether to sort in reverse direction.

    Returns:
        list: [(k, v)].
    """
    index = None
    if by == 'key':
        index = 0
    elif by == 'value':
        index = 1
    else:
        raise ValueError('by only accept `key` or `value`, but got `%s`' % by)
    return sorted(x.items(), key=lambda item: item[index], reverse=reverse)


"""Utils func for squeezing or expanding the first k dim of a tensor"""


def merge_first_k_dim(x, dims):
    """
    Merge the first k dim of a tensor.
    For example:
        >>> x_dims: (a, b, c, d, e)
        >>> x = merge_first_k_dim(x, dims=(0, 1, 2))
        >>> x_dims: [a*b*c, d, e]

    Args:
        x: a tensor has more than 2 dim.
        dims: a tuple or list, containing ascending numbers.

    Returns:
        reshaped x.
    """

    def _check_dims(d):
        assert len(d) >= 2, 'length of dims must greater than 2'
        check_sum = [d[i] - d[i - 1] for i in range(1, len(d))]
        assert len(check_sum) == sum(check_sum), 'dims not continuous'

    _check_dims(dims)
    x_dims = list(x.size())
    remained_dims = [x_dims[i] for i in range(len(x_dims)) if i in dims]
    remained_dims_mul = 1
    for d in remained_dims:
        remained_dims_mul *= d
    x_new_dims = [remained_dims_mul] + [x_dims[i] for i in range(len(x_dims)) if i not in dims]
    return x.reshape(x_new_dims)


def depart_first_dim(x, dims):
    """
        Depart the first dim of a tensor.
        For example:
            >>> x_dims: (a, b, c, d, e)
            >>> a = a1 * a2
            >>> x = depart_first_dim(x, dims=(a1, a2))
            >>> x_dims: [a1, a2, b, c, d, e]

        Args:
            x: a tensor has more than 2 dim.
            dims: a tuple or list, containing ascending numbers.

        Returns:
            reshaped x.
    """
    x_dims = list(x.size())
    x_new_dims = list(dims) + x_dims[1:]
    return x.reshape(x_new_dims)


def get_binary_logits(logits_full, label, base=True):
    """
    Extract binary logits given full logits and the specified labels.

    Args:
        logits_full: the full logits with shape (N, num_classes, h, w).
        label: batch of labels with shape (N).
        base: whether to use base classifiers obtained from training process.

    Returns:
        binary logits with shape (N, 2, h, w) where one channel is for foreground
        and the other is for background.
    """

    label_logits = logits_full[torch.arange(logits_full.size(0), device=label.device), label]
    if base:
        # print("label: ", label)
        other_logits = [logit_k[torch.arange(logit_k.size(0), device=label.device) != label[idx]]
                        for idx, logit_k in enumerate(logits_full)]
        other_logits = torch.stack(other_logits, dim=0)
        other_logits = torch.max(other_logits, dim=1)[0]
    else:
        zeros = torch.zeros_like(label)
        other_logits = logits_full[torch.arange(logits_full.size(0), device=label.device), zeros]
    logits_binary = torch.cat([other_logits.unsqueeze(1), label_logits.unsqueeze(1)], dim=1)
    return logits_binary


if __name__ == "__main__":
    pass
