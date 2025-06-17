from setuptools import setup, find_packages

setup(
    name="card_detector",
    version="0.1.0",
    description="YOLO → CNN card detection pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pytest",
        "pyyaml",
        "Pillow",
        "numpy",
        "torch",
        "torchvision",
        "albumentations",
        "ultralytics",
    ],
    entry_points={
        "console_scripts": [
            "card-detector=card_detector.__main__:main",
            "train-cnn=card_detector.cnn.train:main",
        ],
    },
)
