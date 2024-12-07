from pathlib import Path

HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
TARGET_IMAGE_SIZE = (256, 256)


IMAGE_DATASETS = {
    "real": [
        #{"path": "bitmind/bm-real"},
        #{"path": "bitmind/open-image-v7-256"}, open-image-v7-256
        ##{"path": "bitmind/celeb-a-hq"},
        {"path": "/home/user/.cache/huggingface/celeb-a-hq"},
        #{"path": "bitmind/ffhq-256"},
        #{"path": "bitmind/MS-COCO-unique-256"}
    ],
    "fake": [
        #{"path": "bitmind/bm-realvisxl"},
        ##{"path": "bitmind/bm-mobius-10-17-24"},
        {"path": "/home/user/.cache/huggingface/bm-mobius-10-17-24"},
        #{"path": "bitmind/bm-sdxl"}
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
