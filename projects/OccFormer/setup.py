from setuptools import find_packages, setup


if __name__ == '__main__':
    setup(
        name='OccFormer',
        version='0.0',
        description='OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction',
        author='OccFormer Contributors',
        keywords='3D Occupancy Prediction',
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
        ],
        license="Apache License 2.0",
    )
