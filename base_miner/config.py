from pathlib import Path

HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
TARGET_IMAGE_SIZE = (256, 256)


IMAGE_DATASETS = {
    "real": [
        {"path": "bitmind/bm-real"}, #6.7 GB                28K images
        {"path": "bitmind/open-image-v7-256"}, #216 GB      9 million images
        {"path": "bitmind/celeb-a-hq"}, #2.76 GB,           30K images
        {"path": "bitmind/ffhq-256"}, #7.63 GB              70K images
        {"path": "bitmind/MS-COCO-unique-256"} #13.6 GB     123K images
        {"path": "bitmind/AFHQ"}, #6.96 GB                  15K images
        {"path": "bitmind/lfw"}, #188 MB                    13K images
        {"path": "bitmind/caltech-256"}, #1.2 GB            30K images
        {"path": "bitmind/caltech-101"}, #158 MB            9K images
        {"path": "bitmind/dtd"} #629 MB                     5.6K images (LOWEST IN REAL IMAGES)
    ],
    "fake": [
        {"path": "bitmind/bm-realvisxl"}, #13 GB            10K images
        {"path": "bitmind/bm-mobius"}, #14.3 GB             10K images
        {"path": "bitmind/bm-sdxl"} #12.6 GB                10K images

        # stabilityai/stable-diffusion-xl-base-1.0
         {"path": "bitmind/AFHQ___stable-diffusion-xl-base-1.0"}, # 1.63 GB        15.8K images
         {"path": "bitmind/bm-sdxl"}, # 12.6 GB                                    9.5K images

        # SG161222/RealVisXL_V4.0
         {"path": "bitmind/bm-realvisxl"}, # 13 GB                                  9.7K images

        # Corcelio/mobius
         {"path": "bitmind/bm-mobius"}, # 14.3 GB                                   9.8K images

        # black-forest-labs/FLUX.1-dev
         {"path": "bitmind/bm-subnet-weekly-FLUX.1-dev-256"}, # 76 MB
         {"path": "bitmind/ffhq-256___FLUX.1-dev"}, # 6.65 GB
         {"path": "bitmind/celeb-a-hq___FLUX.1-dev"} # 2.65 GB

        # prompthero/openjourney-v4

        # cagliostrolab/animagine-xl-3.1
       
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
