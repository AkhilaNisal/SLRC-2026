from setuptools import find_packages, setup

package_name = 'apriltag_decoder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'pupil-apriltags',
    ],
    zip_safe=True,
    maintainer='thunderbot',
    maintainer_email='thunderbot@todo.todo',
    description='AprilTag detector/decoder node.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'apriltag_decoder_node = apriltag_decoder.apriltag_decoder_node:main',
        ],
    },
)
