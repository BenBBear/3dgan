import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from plyfile import PlyData, PlyElement
from numpy.random import choice, normal
from functools import reduce
import pickle
import zipfile
from multiprocessing import Pool
from functools import partial
from datetime import datetime
import scipy
from glob import glob
import uuid
from operator import mul
from tqdm import tqdm
import os
from scipy import ndimage
from scipy.io import loadmat, savemat
from math import sqrt
from random import shuffle
from os.path import getsize, dirname, abspath, join, expanduser, isfile, exists, basename
import sys
from six.moves import urllib

MODEL_NET_LABELS = ['bathtub', 'bed', 'chair',
                    'desk', 'dresser', 'monitor', 'nightstand',
                    'sofa', 'table', 'toilet']
bathtub, bed, chair, desk, dresser, monitor, nightstand, sofa, table, toilet = MODEL_NET_LABELS
TRAIN = 'train'
TEST = 'test'


root__ = __root__ = dirname(abspath(__file__))
home__ = __home__ = expanduser("~")
desktop__ = __desktop__ = join(__home__, 'Desktop')
data__ = __data__ = join(__root__, "data")
hardware__ = __hardware__ = join(data__, 'hardware')
celebA_data__ = __celebA_data__ = join("data", "img_align_celeba")
training_ready__ = __training_ready__ = join(__data__, 'training_ready')
model_save_temp_folder__ = __model_save_temp_folder__ = join(
    __data__, "tf_model_temp_save")
log_dir__ = __log_dir__ = join(__data__, 'log_collection')
volume_data_path__ = __volume_data_path__ = join(
    __data__, 'volumetric_data', 'python_pickle')
volume_data_path_matlab__ = __volume_data_path_matlab__ = join(
    __data__, 'volumetric_data', 'volumetric_data_matlab')
inf = 1e99

"""
Misc utilities
"""


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


def make_model_path(model_name):
    return join(model_save_temp_folder__, model_name)


def make_model_images_path(model_name):
    return join(model_save_temp_folder__, model_name, 'images')


def ok():
    print("ok!")


class Log:

    def __init__(self, model_name=None):
        if model_name is not None:
            model_dir = join(__log_dir__, model_name)
            mkdir(model_dir)
            self.log_file = open(join(model_dir, 'training.log'), 'w')
        else:
            self.log_file = None

    def __call__(self, message):
        text = "LOG: " + message
        print(text)
        if self.log_file is not None:
            print(text, file=self.log_file)


def pause():
    input("Program stop, press the <ENTER> key to continue...")


def exception_if(condition, message):
    if condition:
        raise Exception(message)


def make_id(length=8):
    return str(uuid.uuid4())[:length]


def identity_single(a):
    return a


def identity_many(*args):
    return args


def list_prod(lst, dtype=int):
    return reduce(mul, [dtype(l) for l in lst], 1)


def to_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x, ]

"""
List and Numpy utilities
"""


def flatten_list(lst):
    return sum(lst, [])


def numpy_to_tuple_list(arr):
    return [tuple(arr[i, :]) for i in range(arr.shape[0])]


def pad_numpy_array_on_end(a, padding=None):
    if padding is None:
        return a
    else:
        if padding < a.shape[0]:
            raise Exception("Padding = {0}, however the array need to pad has size = {1}"
                            .format(padding, a.shape[0]))
        else:
            r = np.zeros([padding] + list(a.shape[1:]))
            r[:a.shape[0], :] = a
            return r


def concat_numpy_array_list(lst):
    if isinstance(lst, np.ndarray):
        return lst
    shape = list(lst[0].shape)
    shape[:0] = [len(lst)]
    return np.concatenate(lst).reshape(shape)


def numpy_array_spiking(array, threshold=0.5, maxval=1, minval=0, copy=False):
    max_idx = array >= threshold
    min_idx = array < threshold
    array = array if not copy else np.copy(array)
    array[max_idx] = maxval
    array[min_idx] = minval
    return array

"""
Visualization utilities
"""


def to_ply(dct, path):
    verts = np.array(numpy_to_tuple_list(dct['verts']),
                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces = np.array([(list(it),) for it in numpy_to_tuple_list(dct['faces'])],
                     dtype=[('vertex_indices', 'i4', (3,))])
    plydata = PlyData(
        [
            PlyElement.describe(verts, 'vertex'),
            PlyElement.describe(faces, 'face')
        ], text=True)
    plydata.write(path)


def triangle_numpy_array_to_ply(np_array, path=join(__desktop__, 'output.ply'), shape=(3, 3)):
    seq = np_array.reshape([np_array.shape[0]] + list(shape))
    verts = seq.reshape(seq.shape[0] * 3, seq.shape[2])
    faces = np.arange(seq.shape[0] * 3).reshape((seq.shape[0], 3))
    data = {
        'verts': verts,
        'faces': faces
    }
    to_ply(data, path)


def plot_loss(loss,  save_to=None, title="Loss Plotting"):
    from IPython import display
    import matplotlib.pyplot as plt
    # plt.switch_backend('agg')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    for l in loss:
        plt.plot(l[1], label=l[0])
    plt.legend()
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
    plt.show()


def plot_grid_image(gen_images, image_shape=None, dim=(8, 8), figure_size=(12, 12), show=True, save_to=None):
    import matplotlib.pyplot as plt
    # plt.switch_backend('agg')
    plt.figure(figsize=figure_size)
    for i, image in enumerate(gen_images):
        if i >= list_prod(dim):
            break
        plt.subplot(dim[0], dim[1], i + 1)
        if image_shape is not None:
            image = image.reshape(image_shape)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()


def plot_volume(v, save_to=None, show=True):
    import matplotlib.pyplot as plt
    # plt.switch_backend('agg')
    from mpl_toolkits.mplot3d import Axes3D
    z, x, y = v.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red', marker='o')
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()


def plot_grid_volume(gen_volumes, dim=(8, 8), figure_size=(30, 30), volume_shape=None, show=True, save_to=None):
    import matplotlib.pyplot as plt
    # plt.switch_backend('agg')
    from mpl_toolkits.mplot3d import Axes3D
    plt.figure(figsize=figure_size)
    for i, volume in enumerate(gen_volumes):
        if i >= list_prod(dim):
            break
        if volume_shape is not None:
            volume = volume.reshape(volume_shape)
        ax = plt.subplot(dim[0], dim[1], i + 1, projection='3d')
        z, x, y = volume.nonzero()
        ax.scatter(x, y, -z, zdir='z', c='red', marker='o')
        plt.axis('off')
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
    if show:
        plt.show()
"""
Training Utilities
"""


def pickle_dump(data, file_path):
    with open(file_path, 'wb') as handle:
        bytes_out = pickle.dumps(data, protocol=2)
        n_bytes = len(bytes_out)
        max_bytes = 2 ** 31 - 1
        for idx in range(0, n_bytes, max_bytes):
            handle.write(bytes_out[idx:idx + max_bytes])


def pickle_load(file_path):
    bytes_in = bytearray(0)
    max_bytes = 2 ** 31 - 1
    input_size = getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


@static_var("counter", 0)
def get_batch_indexes(size, bs, shuffle=True, reset=False):
    if reset:
        get_batch_indexes.counter = 0
    if shuffle:
        pts = choice(size, bs, replace=False)
    else:
        c = get_batch_indexes.counter
        if c + bs < size:
            pts = np.arange(c, c + bs)
        else:
            pts = np.concatenate(
                (np.arange(c, size), np.arange(0, (c + bs) % size)))
        get_batch_indexes.counter += bs
        get_batch_indexes.counter %= size
    return pts.astype(np.int32)


def mkdir(*paths):
    for path in paths:
        if not exists(path):
            os.makedirs(path)


def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))
    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
                  ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath


def download_celeb_a(dirpath, data_dir='img_align_celeba'):
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return
    url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        print("unzipping")
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
        print("finished")
    print("cleaning")
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))


def filelist(path, shuffled=False):
    result = [os.path.realpath(p) for p in glob(path)]
    if shuffled:
        shuffle(result)
    return result


def shift_sequence(array, shift_by=1):
    """
    :param array: [batch_size, sequence, 9]
    :param shift_by:
    :return:
    """
    result = np.zeros(array.shape)
    result[:, shift_by:, :] = array[:, :-shift_by, :]
    return result


def get_training_ready_data_set(label="table", max_seq_length=1000, padding=None,
                                order_list=None):
    if order_list is None:
        order_list = [TRIANGULAR_SEQUENCE_ORDER_AREA,
                      TRIANGULAR_SEQUENCE_ORDER_ORIGIN_AVG_DISTANCE,
                      TRIANGULAR_SEQUENCE_ORDER_Z]
    ready_data = {'label': label}
    for order in order_list:
        order_name = TRIANGULAR_SEQUENCE_ORDER_NAME[order]
        base_path_name = join(__root__, "data/model_net_10/modelnet"
                              "10.grid-basic-3x3.{0}.{1}.{2}.tseq.pickle"
                              .format(order_name, label, "{0}"))
        _data = pickle_load(base_path_name.format(TRAIN)) + \
            pickle_load(base_path_name.format(TEST))
        _data_with_max_length = [d['sequence']
                                 for d in _data if len(d['sequence']) <= max_seq_length]
        _data_with_triangle_vertex_sorted = []
        for d in _data_with_max_length:
            _data_with_triangle_vertex_sorted.append(concat_numpy_array_list(
                [sort_triangle_vertexes_by_origin_distance(_d) for _d in d]))
        _data_with_max_length_Nx9 = [d.reshape([d.shape[0]] + [d.shape[1] * d.shape[2]])
                                     for d in _data_with_triangle_vertex_sorted]
        _data_with_max_length_Nx9_padding = _data_with_max_length_Nx9
        if padding:
            _data_with_max_length_Nx9_padding = [pad_numpy_array_on_end(d, padding=max_seq_length)
                                                 for d in _data_with_max_length_Nx9]
            ready_data[order_name] = concat_numpy_array_list(
                _data_with_max_length_Nx9_padding)
        ready_data[order_name] = _data_with_max_length_Nx9_padding
        ready_data['padding'] = padding
        ready_data['len'] = len(_data_with_max_length_Nx9_padding)

    # Need to Check indexes are match for each order!
    return ready_data


def triangle_area(a, b, c):
    ab, ac = a - b, a - c
    n = (np.linalg.norm(ab) * np.linalg.norm(ac))
    cos_theta = np.dot(ab, ac) / n
    sin_pow = abs(1 - cos_theta**2)
    sin_theta = sqrt(sin_pow)
    return 0.5 * sin_theta * n


def sort_triangle_vertexes_by_origin_distance(d):
    """
    :param d:  3x3 numpy array
    :return: 3x3 numpy array, sort by origin distance on the axis 0
    """
    # 但是事实证明， 三角形order好像有关系！！
    # dist = np.linalg.norm(d, axis=1)
    # ind = np.argsort(dist)
    # return d[ind, :]
    return d


def sort_batch_triangle_numpy_array(array, order_mode, shape=(3, 3)):
    return concat_numpy_array_list([sort_single_triangle_numpy_array(a,
                                                                     order_mode=order_mode,
                                                                     shape=shape) for a in array])


def sort_single_triangle_numpy_array(array, order_mode, shape=(3, 3)):
    """
    assume array is a numpy (size x 9),
    return same shape numpy array but sorted
    """
    sequence = [a.reshape(shape) for a in array]

    def _triangle_avg_point(t):
        return t.mean(axis=0)
    if order_mode == TRIANGULAR_SEQUENCE_ORDER_AREA:
        sequence = sorted(sequence, key=lambda x: triangle_area(
            x[0, :], x[1, :], x[2, :]))
    elif order_mode == TRIANGULAR_SEQUENCE_ORDER_DEGREE:
        raise NotImplementedError(
            "Order Mode:{0} is not implemented".format(order_mode))
    elif order_mode == TRIANGULAR_SEQUENCE_ORDER_ORIGIN_AVG_DISTANCE:
        sequence = sorted(
            sequence, key=lambda x: np.linalg.norm(_triangle_avg_point(x)))
    elif order_mode == TRIANGULAR_SEQUENCE_ORDER_ORIGIN_MAX_DISTANCE:
        sequence = sorted(sequence, key=lambda x: x.max())
    elif order_mode == TRIANGULAR_SEQUENCE_ORDER_ORIGIN_MIN_DISTANCE:
        sequence = sorted(sequence, key=lambda x: x.min())
    elif order_mode == TRIANGULAR_SEQUENCE_ORDER_Z:
        sequence = concat_numpy_array_list(sequence)
        seqmean = sequence.mean(axis=1)
        ind = np.lexsort((seqmean[:, 1], seqmean[:, 0], seqmean[:, 2]), axis=0)
        sequence = sequence[ind, :, :]
    elif order_mode == TRIANGULAR_SEQUENCE_ORDER_X:
        sequence = concat_numpy_array_list(sequence)
        seqmean = sequence.mean(axis=1)
        ind = np.lexsort((seqmean[:, 2], seqmean[:, 1], seqmean[:, 0]), axis=0)
        sequence = sequence[ind, :, :]
    elif order_mode == TRIANGULAR_SEQUENCE_ORDER_Y:
        sequence = concat_numpy_array_list(sequence)
        seqmean = sequence.mean(axis=1)
        ind = np.lexsort((seqmean[:, 2], seqmean[:, 0], seqmean[:, 1]), axis=0)
        sequence = sequence[ind, :, :]
    else:
        raise NotImplementedError(
            "Order Mode:{0} is not implemented".format(order_mode))
    return sequence.reshape([sequence.shape[0], sequence.shape[1] * sequence.shape[2]])


def convert_mat_list_for_class(arg, class_name):
    base_str, output_dir, vs, angle = arg
    volume_mat_filename_list = filelist(base_str.format(class_name))
    volume_np_list = []
    label_np_list = []
    if angle == 'all':
        angle = [i for i in range(1, 13)]
    for p in tqdm(volume_mat_filename_list, desc="Processing_{0}".format(class_name)):
        angle_label = int(basename(p).split('.')[0].split('_')[-1])
        if angle_label in angle:
            volume_np_list.append(loadmat(p)['instance'])
            label_np_list.append(angle_label - 1)
    if len(volume_np_list) is 0:
        return
    angle_str = ",".join([str(i) for i in angle])
    pickle_dump({
        'data': concat_numpy_array_list(volume_np_list),
        'label': np.array(label_np_list)
    }, join(output_dir, '{0}_{1}_angle_{2}.pickle'.format(class_name, vs, angle_str)))


def convert_volume_mat_to_pickle(volume_mat_input_dir, volume_pickle_output_dir,
                                 class_name_list=['chair'], volume_size=30, num_threads=1, angle='all'):
    volume_matching_str = join(
        volume_mat_input_dir, '{0}/{1}/**/*.mat'.format('{0}', volume_size))
    _convert_partial = partial(
        convert_mat_list_for_class, [volume_matching_str, volume_pickle_output_dir, volume_size, angle])
    pool = Pool(processes=num_threads)
    for _ in pool.imap_unordered(_convert_partial, class_name_list):
        pass
    pool.close()


def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y) + 1
    x = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        x[i, y[i]] = 1
    return x


def soften_categorical_labels(y, l=0.1, h=0.9):
    y = y.astype(np.float32)
    if l >= h:
        raise Exception(
            "soften_categorical_labels: l must be smaller than h! ")
    y *= (h - l)
    y += l
    return y


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])


def image_to_rgb(im, dtype=np.float32):
    if len(im.shape) > 2:
        if im.shape[2] > 3:
            return im[:, :, :3]
        else:
            return im
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=dtype)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


def transform(image, npx=64, is_crop=True, resize_w=64, is_grayscale=False):
    # npx : # of pixels width/height of image
    if not is_grayscale:
        image = image_to_rgb(image)
    image = pad_image(image, npx)
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.
image_transform = transform


def inverse_transform(images):
    return (images + 1.) / 2.
image_inverse_transform = inverse_transform


def imread(path, is_grayscale=False):
    if is_grayscale:
        return np.asarray(ndimage.imread(path, flatten=True)).astype(np.float32)
    else:
        return np.asarray(ndimage.imread(path)).astype(np.float32)


def pad_image(image, size):
    if isinstance(size, int):
        size = (size, size)
    s1, s2 = size
    i1, i2, c = image.shape
    if s1 > i1:
        temp = np.zeros((s1, i2, c))
        start = int((s1 - i1) / 2)
        temp[start:start + i1, :, :] = image
        image = temp
        i1 = s1
    if s2 > i2:
        temp = np.zeros((i1, s2, c))
        start = int((s2 - i2) / 2)
        temp[:, start:start + i2, :] = image
        image = temp
    return image


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w,
                     is_grayscale=is_grayscale)
