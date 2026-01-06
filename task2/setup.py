from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'task2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch_folder'), glob('launch_folder/*.py')),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.rviz')),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.stl')),
    ],  
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mirco',
    maintainer_email='your_email@example.com',
    description='Aggregative tracking formation control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aggregative_agent = task2.the_agent:main',
            'task2_visualizer = task2.task2_visualizer:main',
            'task2_plotter = task2.task2_plotter:main',
        ],
    },
)