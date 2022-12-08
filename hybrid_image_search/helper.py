"""Helper module for Pinecone's image search example"""

import os
import itertools
import re
import getpass
import random
import textwrap

from IPython.display import display, Markdown
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns

from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor,
    Normalize
)


ENVIRONMENTAL_VARIABLE_NAME = 'PINECONE_EXAMPLE_API_KEY'
PINECONE_PALETTE = [
    "#1C17FF",
    "#000000",
    "#030080",
    "#25239D",
    "#738FAB",
    "#DFECF9",
    "#F1F5F8",
    "#FFFFFF",
    "#FAFF00",
    "#8CF1FF"
]
pinecone_api_key = None


printmd = lambda x: display(Markdown(x))


def set_pinecone_api_key():
    global pinecone_api_key
    api_key_prompt = (
        f'{ENVIRONMENTAL_VARIABLE_NAME} not found in environmental variables list.\n'
        'Get yours at https://app.pinecone.io and enter it here: '
    )
    printmd(f'Extracting API Key from environmental variable `{ENVIRONMENTAL_VARIABLE_NAME}`...')
    pinecone_api_key = os.getenv(ENVIRONMENTAL_VARIABLE_NAME)
    if not pinecone_api_key:
        printmd(api_key_prompt)
        pinecone_api_key = getpass.getpass('')
    printmd('Pinecone API Key available at `h.pinecone_api_key`')


preprocess = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def format_cpations_text(captions, num_captions=3, wrap_len=60):
    captions = [_ for cap in captions[:num_captions] for _ in textwrap.wrap("- " + cap, wrap_len)]
    return "\n".join(captions)

def normalize_float_image(img):
    img -= np.amin(img)
    img = img / np.amax(img)
    return img


def show_random_images_from_full_dataset(dset, num_rows=4, num_cols=2):
    """Show random sample of images in PyTorch dataset."""

    ### get random sample of images and labels
    indices = np.random.randint(0, high=len(dset) + 1, size=num_rows * num_cols)

    ### plot sample
    fig = plt.figure(figsize=(30, 30))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(num_rows, num_cols),
        axes_pad=1)
    for ax, idx in zip(grid, indices):
        im_array, caption = dset[idx]
        ax.imshow(normalize_float_image(im_array.permute(1, 2, 0).numpy()))
        ax.set_title(format_cpations_text(caption))
        ax.axis("off")


def get_tqdm_kwargs(dataloader, num_rows=np.inf):
    batch_size, total_samples = dataloader.batch_size, min(len(dataloader.dataset), num_rows)
    return dict(
        smoothing=0, 
        unit=f'chunk of {batch_size} '
             f'{dataloader.dataset.__class__.__name__} vectors',
        total=int(np.ceil(total_samples/batch_size))
    )


def _get_ids_scores_metadatas(response):
    """Return ids and scores from Pinecone query response."""
    matches = response.to_dict()['matches']
    ids, scores, metadatas = zip(*[(
        match['id'], 
        match['score'], 
        match['metadata'] if 'metadata' in match else {}
    ) for match in matches])
    return list(ids), list(scores), list(metadatas)


def get_response_information(response):
    """Return dataset, ids, and scores from Pinecone query response."""
    ids, scores, metadatas = _get_ids_scores_metadatas(response)
    datasets, rows = list(zip(*[id_.split('.') for id_ in ids]))
    return datasets, map(int, rows), scores, metadatas


def show_response_as_grid(response, dataset, nrows=2, ncols=2, 
                          num_captions=3, wrap_len=60, 
                          axes=None, **subplot_kwargs):
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, **subplot_kwargs)
    # fig.tight_layout()
    iter_response = get_response_information(response)
    iter_images = zip(*[*iter_response, axes.flat])
    for dataset_name, row, score, metadata, ax in iter_images:
        image, caption = dataset[row]
        image = image.permute(1, 2, 0).numpy()
        ax.imshow(normalize_float_image(image))
        ax.set_title(
            format_cpations_text(caption, num_captions=num_captions, wrap_len=wrap_len) + f"\nScore: {score : .3f}"
        )
        ax.axis("off")


def show_query_image(query_image, query_captions):
    fig = plt.figure(figsize=(3.5,3.5))
    img = normalize_float_image(query_image.permute(1, 2, 0).numpy())
    plt.imshow(img)
    plt.title("Keywords: " + format_cpations_text(query_captions)[2:], fontsize=14)
    fig.text(0.5, 1, f"Query Image", horizontalalignment='center', fontsize=16)
    plt.axis('off')

def show_query_image_and_keywords(query_image, keywords):
    fig = plt.figure(figsize=(4 ,4))
    img = normalize_float_image(query_image.permute(1, 2, 0).numpy())
    plt.imshow(img)
    plt.title("Keywords: " + keywords, fontsize=14)
    fig.text(0.5, 1, f"Query Image", horizontalalignment='center', fontsize=16)
    plt.axis('off')


def run_on_module_import():
    sns.set_palette(PINECONE_PALETTE)
    set_pinecone_api_key()
    import warnings
    warnings.filterwarnings('ignore')


run_on_module_import()
