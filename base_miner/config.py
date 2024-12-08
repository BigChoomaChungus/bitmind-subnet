from pathlib import Path

HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
TARGET_IMAGE_SIZE = (256, 256)


IMAGE_DATASETS = {
    "real": [
        {"path": "bitmind/bm-real"}, #6.7 GB
        #{"path": "bitmind/open-image-v7-256"}, open-image-v7-256
        #{"path": "bitmind/celeb-a-hq"},
        #{"path": "/home/user/.cache/huggingface/bitmind___celeb-a-hq/default/0.0.0/cec732e87335bc65872d90a706e865032daf80fc"},
        #{"path": "bitmind/ffhq-256"},
        #{"path": "bitmind/MS-COCO-unique-256"} #13.6 GB
    ],
    "fake": [
        #{"path": "bitmind/bm-realvisxl"}, #13 GB
        #{"path": "bitmind/bm-mobius-10-17-24"}, #1.83 GB
        #{"path": "/home/user/.cache/huggingface/bitmind___bm-mobius-10-17-24/default/0.0.0/dc8eb1065fb36e48a03e9cff6824cde003bcece5"},
        {"path": "bitmind/bm-sdxl"} #12.6 GB
    ]
}

# see bitmind-subnet/create_video_dataset_example.sh 
VIDEO_DATASETS = {
    "real": [
        {"path": ""}
    ],
    "fake": [
        {"path": ""}
    ]
}

FACE_IMAGE_DATASETS = {
    "real": [
        {"path": "bitmind/ffhq-256_training_faces", "name": "base_transforms"},
        {"path": "bitmind/celeb-a-hq_training_faces", "name": "base_transforms"}

    ],
    "fake": [
        {"path": "bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces", "name": "base_transforms"},
        {"path": "bitmind/celeb-a-hq___stable-diffusion-xl-base-1.0___256_training_faces", "name": "base_transforms"}
    ]
}
