import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import torch
import warnings
import zipfile

from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def imshow(image, click_event=None, cmap=None):
    """
    Show an image at true scale
    """
    import matplotlib.pyplot as plt
    dpi = 80
    margin = 0.5  # (5% of the width/height of the figure...)
    h, w = image.shape[:2]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * w / dpi, (1 + margin) * h / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(image, interpolation='none', cmap=cmap)

    plt.axis('off')
    plt.show()

    return fig, ax


def get_sed_model_file():
    import os.path
    default_path = 'data/opencv_sed_model.yml.gz'
    if os.path.isfile(default_path):
        return default_path
    else:
        import urllib.request as request
        dl_path = './opencv_sed_model.yml.gz'
        request.urlretrieve('https://github.com/higra/Higra-Notebooks/raw/master/data/opencv_sed_model.yml.gz', dl_path)
        return dl_path


def locate_resource(name):
    import os.path
    default_path = 'data/' + name
    if os.path.isfile(default_path):
        return default_path
    else:
        return 'https://github.com/higra/Higra-Notebooks/raw/master/data/' + name


def enable_plotly_in_cell():
    """
    To be used in colab: this method pre-populates the outputframe with the configuration that Plotly expects 
    and must be executed for every cell which is displaying a Plotly graph.
    
    https://colab.research.google.com/notebooks/charts.ipynb#scrollTo=niTJd49yO4xf
    """
    import IPython
    from plotly.offline import init_notebook_mode
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
    '''))
    init_notebook_mode(connected=False)


def saturate_max(array, max_saturation=0.005, normalize=True):
    import numpy as np
    values = np.ravel(array)

    sorted_values = np.sort(values)
    max_v = sorted_values[min(len(sorted_values) - 1, int((1 - max_saturation) * len(sorted_values)))]
    saturated_im = array.copy()
    saturated_im[saturated_im > max_v] = max_v

    if normalize:
        minv = np.min(saturated_im)
        maxv = np.max(saturated_im)
        saturated_im = (saturated_im - minv) / (maxv - minv)

    return saturated_im


def get_tile_images(image, width, height):
    import numpy as np
    _nrows, _ncols = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        image = image[:-_m, :-_n]

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width),
        strides=(width * _strides[1], height * _strides[0], *_strides),
        writeable=False
    )


def imread(filename):
    """
    Simple wrapper around imageio.imread to avoid issues with image reading from url
    see note at https://imageio.readthedocs.io/en/stable/userapi.html
    """
    import imageio
    im = None
    if filename.startswith("http"):
        dotposition = filename.rfind(".")
        if dotposition != -1:
            extension = filename[dotposition:]
            im = imageio.imread(imageio.core.urlopen(filename).read(), extension)
    if im is None:
        im = imageio.imread(filename)
    return im
