from setuptools import find_packages, setup

package_name = 'coverage_client'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nesl',
    maintainer_email='guanhuaji@g.ucla.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coverage = coverage_client.coverage:main',
            'localization = coverage_client.object_localization:main'
        ],
    },
)
