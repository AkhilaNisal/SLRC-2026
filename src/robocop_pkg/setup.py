import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robocop_pkg'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/robocop_launch.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/robocop_launch1.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/robocop_launchnew.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/robocop_launch_obstacle.py']))


data_files.append(('share/' + package_name + '/worlds', ['worlds/my_world.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/arena.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/arena1.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/arena2.wbt']))

proto_files = glob('protos/*.proto')
if proto_files:
    data_files.append((f'share/{package_name}/protos', proto_files))

# data_files.append(('share/' + package_name + '/protos', ['protos/robocop.proto']))
data_files.append(('share/' + package_name + '/resource', ['resource/robocop.urdf']))
data_files.append(('share/' + package_name, ['package.xml']))
data_files.append((
    'share/' + package_name + '/resource/arena_meshes',
    glob('resource/arena_meshes/*.dae')
))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akhila-wedamestrige',
    maintainer_email='wedamestrigean@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robocop_driver = robocop_pkg.robocop_driver:main',
            'obstacle_avoider = robocop_pkg.obstacle_avoider:main',
            'obstacle_seeker = robocop_pkg.obstacle_seeker:main',
            'white_line_follower = robocop_pkg.white_line_follower:main',
            'red_box_seeker = robocop_pkg.red_box_seeker:main',
        ],
    },
)
