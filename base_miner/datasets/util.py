from typing import List, Tuple, Dict
import torchvision.transforms as transforms
import numpy as np
import datasets
from torch.utils.data import Subset

from base_miner.datasets.download_data import load_huggingface_dataset
from base_miner.datasets import ImageDataset, VideoDataset, RealFakeDataset

datasets.logging.set_verbosity_error()
datasets.disable_progress_bar()


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

class PatchedSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy necessary attributes from the original dataset
        if hasattr(dataset, "huggingface_dataset_path"):
            self.huggingface_dataset_path = dataset.huggingface_dataset_path

    def __getattr__(self, attr):
        # If an attribute is not found in Subset, fallback to the original dataset
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

def create_real_fake_datasets(
    real_datasets: Dict[str, List[ImageDataset]],
    fake_datasets: Dict[str, List[ImageDataset]],
    source_labels: bool = False,
    group_sources_by_name: bool = False) -> Tuple[RealFakeDataset, ...]:
    """
    Adjust dataset balancing logic to handle Subset properly.
    """
    source_label_mapping = None
    if source_labels:
        source_label_mapping = create_source_label_mapping(
            real_datasets, fake_datasets, group_sources_by_name)

    # Calculate total sizes for real and fake datasets
    real_train_datasets = real_datasets['train']
    fake_train_datasets = fake_datasets['train']
    real_total_size = sum(len(dataset) for dataset in real_train_datasets)
    fake_total_size = sum(len(dataset) for dataset in fake_train_datasets)

    # Use the smaller total size for balancing
    min_size = min(real_total_size, fake_total_size)

    print(f"Total real dataset size: {real_total_size}")  # Debug line
    print(f"Total fake dataset size: {fake_total_size}")  # Debug line
    print(f"Balanced dataset size: {min_size}")  # Debug line

    # Trim datasets to the balanced size and preserve original attributes
    def create_balanced_subsets(datasets, total_count):
        balanced_datasets = []
        count = 0

        for dataset in datasets:
            if count >= total_count:
                break
            remaining = total_count - count
            subset_indices = list(range(min(len(dataset), remaining)))

            # Preserve original dataset attributes
            if hasattr(dataset, "huggingface_dataset_path"):
                dataset.huggingface_dataset_path = getattr(dataset, "huggingface_dataset_path")

            balanced_datasets.append(Subset(dataset, subset_indices))
            count += len(subset_indices)

        return balanced_datasets

    balanced_real_datasets = create_balanced_subsets(real_train_datasets, min_size)
    balanced_fake_datasets = create_balanced_subsets(fake_train_datasets, min_size)

    train_dataset = RealFakeDataset(
        real_image_datasets=balanced_real_datasets,
        fake_image_datasets=balanced_fake_datasets,
        source_label_mapping=source_label_mapping,
    )

    val_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['validation'],
        fake_image_datasets=fake_datasets['validation'],
        source_label_mapping=source_label_mapping,
    )

    test_dataset = RealFakeDataset(
        real_image_datasets=real_datasets['test'],
        fake_image_datasets=fake_datasets['test'],
        source_label_mapping=source_label_mapping,
    )

    if source_labels:
        return train_dataset, val_dataset, test_dataset, source_label_mapping
    return train_dataset, val_dataset, test_dataset


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
