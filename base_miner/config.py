from pathlib import Path

HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
TARGET_IMAGE_SIZE = (256, 256)


IMAGE_DATASETS = {
    "real": [
        {"path": "bitmind/bm-real"}, #6.7 GB
        {"path": "bitmind/open-image-v7-256"}, #216 GB
        {"path": "bitmind/celeb-a-hq"}, #2.76 GB
        {"path": "bitmind/ffhq-256"}, #7.63 GB
        {"path": "bitmind/MS-COCO-unique-256"} #13.6 GB
        {"path": "bitmind/AFHQ"}, #6.96 GB
        {"path": "bitmind/lfw"}, #188 MB
        {"path": "bitmind/caltech-256"}, #1.2 GB
        {"path": "bitmind/caltech-101"}, #158 MB
        {"path": "bitmind/dtd"} #629 MB
    ],
    "fake": [
        {"path": "bitmind/bm-realvisxl"}, #13 GB
        {"path": "bitmind/bm-mobius"}, #14.3 GB
        {"path": "bitmind/bm-sdxl"} #12.6 GB
        
        #{"path": "bitmind/bm-mobius-10-17-24"}, #1.83 GB #I ADDED THIS ONE BECAUSE ITS SMALL AND I USE IT FOR DEBUGGING, DO NOT USE THIS FOR REAL TRAINING
       
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
