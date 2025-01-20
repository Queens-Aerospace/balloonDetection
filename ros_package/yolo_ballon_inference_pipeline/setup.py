from setuptools import setup

package_name = 'yolo_ballon_inference_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hjenkins',
    maintainer_email='hjenkins@example.com',
    description='YOLO inference pipeline with ROS 2 integration',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_ballon_inference_pipeline = yolo_ballon_inference_pipeline.yolo_ballon_inference_pipeline:main',
        ],
    },
)
