from typing import List, Tuple, Dict
import torchvision.transforms as transforms
import numpy as np
import datasets
from torch.utils.data import Subset

from base_miner.datasets.download_data import load_huggingface_dataset
from base_miner.datasets import ImageDataset, VideoDataset, RealFakeDataset

datasets.logging.set_verbosity_error()
datasets.disable_progress_bar()

class PatchedSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy necessary attributes from the original dataset
        if hasattr(dataset, "huggingface_dataset_path"):
            self.huggingface_dataset_path = dataset.huggingface_dataset_path
        if hasattr(dataset, "_history"):
            self._history = dataset._history

    def __getattr__(self, attr):
        # If an attribute is not found in Subset, fallback to the original dataset
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        


def create_real_fake_datasets(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    source_labels: bool = False,
    group_sources_by_name: bool = False,
    balanced_sizes: Dict[str, int] = None  # New argument for balanced sizes
) -> Tuple[RealFakeDataset, ...]:
    """
    Args:
        real_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets.
        fake_datasets: Dict containing train, val, and test keys. Each key maps to a list of ImageDatasets.
        source_labels: Whether to include source labels for datasets.
        group_sources_by_name: Whether to group fake sources by their model names.
        balanced_sizes: Pre-calculated balanced sizes for each split (optional).

    Returns:
        Train, validation, and test RealFakeDatasets (or optionally with source labels).
    """
    source_label_mapping = None
    if source_labels:
        source_label_mapping = create_source_label_mapping(
            real_datasets, fake_datasets, group_sources_by_name)

    print(f"Source label mapping: {source_label_mapping}")

    datasets = {}
    for split in ['train', 'validation', 'test']:
        if balanced_sizes:
            # If balanced sizes are provided, balance the datasets based on the sizes
            balanced_real_datasets = create_balanced_subsets(real_datasets[split], balanced_sizes[split])
            balanced_fake_datasets = create_balanced_subsets(fake_datasets[split], balanced_sizes[split])
        else:
            # If no balanced sizes, use the datasets as they are
            balanced_real_datasets = real_datasets[split]
            balanced_fake_datasets = fake_datasets[split]

        # Debugging: Print the sizes after balancing
        print(f"{split.capitalize()} balanced real dataset sizes: {[len(ds) for ds in balanced_real_datasets]}")
        print(f"{split.capitalize()} balanced fake dataset sizes: {[len(ds) for ds in balanced_fake_datasets]}")

        datasets[split] = RealFakeDataset(
            real_image_datasets=balanced_real_datasets,
            fake_image_datasets=balanced_fake_datasets,
            source_label_mapping=source_label_mapping
        )

    if source_labels:
        return (
            datasets['train'],
            datasets['validation'],
            datasets['test'],
            source_label_mapping
        )
    return datasets['train'], datasets['validation'], datasets['test']


def create_balanced_subsets(datasets: List[ImageDataset], max_dataset_size: int) -> List[ImageDataset]:
    """
    Balances a list of datasets by limiting their combined size to `max_dataset_size`.
    """
    total_size = sum(len(dataset) for dataset in datasets)
    balanced_datasets = []

    if total_size <= max_dataset_size:
        # If the total size is already smaller than max, return the datasets as is
        return datasets

    remaining = max_dataset_size
    for dataset in datasets:
        if remaining <= 0:
            break

        if len(dataset) > remaining:
            # Take only a subset of this dataset to fit the remaining size
            balanced_datasets.append(dataset[:remaining])
            remaining = 0
        else:
            # Use the full dataset
            balanced_datasets.append(dataset)
            remaining -= len(dataset)

    return balanced_datasets















def split_dataset(dataset):
    # Split data into train, validation, test and return the three splits
    dataset = dataset.shuffle(seed=42)

    if 'train' in dataset:
        dataset = dataset['train']

    split_dataset = {}
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    split_dataset['train'] = train_test_split['train']
    temp_dataset = train_test_split['test']

    # Split the temporary dataset into validation and test
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    split_dataset['validation'] = val_test_split['train']
    split_dataset['test'] = val_test_split['test']

    return split_dataset['train'], split_dataset['validation'], split_dataset['test']


def load_and_split_datasets(
    dataset_meta: list,
    modality: str,
    split_transforms: Dict[str, transforms.Compose] = {},
) -> Dict[str, List[ImageDataset]]:
    """
    Helper function to load and split dataset into train, validation, and test sets.

    Args:
        dataset_meta: List containing metadata about the dataset to load.

    Returns:
        A dictionary with keys == "train", "validation", or "test" strings,
        and values == List[ImageDataset].

        Dict[str, List[ImageDataset]]
    """
    splits = ['train', 'validation', 'test']
    datasets = {split: [] for split in splits}

    for meta in dataset_meta:
        dataset = load_huggingface_dataset(meta['path'], None, meta.get('name'), download_mode="reuse_dataset_if_exists")
        train_ds, val_ds, test_ds = split_dataset(dataset)
        print(f"FROM UTIL.PY: Loaded dataset: {meta['path']}")  # TEST LINE
        print(f"Train split size: {len(train_ds)}")  # TEST LINE
        print(f"Validation split size: {len(val_ds)}")  # TEST LINE
        print(f"Test split size: {len(test_ds)}")  # TEST LINE

        for split, data in zip(splits, [train_ds, val_ds, test_ds]):
            if modality == 'image':
                image_dataset = ImageDataset(huggingface_dataset=data, transforms=split_transforms.get(split, None))
            elif modality == 'video':
                image_dataset = VideoDataset(huggingface_dataset=data, transforms=split_transforms.get(split, None))
            else:
                raise NotImplementedError(f'Unsupported modality: {modality}')
            datasets[split].append(image_dataset)

        split_lengths = ', '.join([f"{split} len={len(datasets[split][0])}" for split in splits])
        print(f'done, {split_lengths}')

    return datasets


def create_source_label_mapping(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    group_by_name: bool = False
    ) -> Dict:

    source_label_mapping = {}
    grouped_source_labels = {}
    # Iterate through real datasets and set their source label to 0.0
    for split, dataset_list in real_datasets.items():
        for dataset in dataset_list:
            source = dataset.huggingface_dataset_path
            if source not in source_label_mapping.keys():
                source_label_mapping[source] = 0.0

    # Assign incremental labels to fake datasets
    for split, dataset_list in fake_datasets.items():
        for dataset in dataset_list:
            source = dataset.huggingface_dataset_path
            if group_by_name and '__' in source:
                model_name = source.split('__')[1]
                if model_name in grouped_source_labels:
                    fake_source_label = grouped_source_labels[model_name]
                else:
                    fake_source_label = max(source_label_mapping.values()) + 1
                    grouped_source_labels[model_name] = fake_source_label

                if source not in source_label_mapping:
                    source_label_mapping[source] = fake_source_label
            else:
                if source not in source_label_mapping:
                    source_label_mapping[source] = max(source_label_mapping.values()) + 1

    return source_label_mapping


def sample_dataset_index_name(image_datasets: list) -> tuple[int, str]:
    """
    Randomly selects a dataset index from the provided dataset list and returns the index and source name.

    Parameters
    ----------
    image_datasets : list
        A list of dataset objects to select from.

    Returns
    -------
    tuple[int, str]
        A tuple containing the index of the randomly selected dataset and the source name.
    """
    dataset_index = np.random.randint(0, len(image_datasets))
    source_name = image_datasets[dataset_index].huggingface_dataset_path
    return dataset_index, source_name
