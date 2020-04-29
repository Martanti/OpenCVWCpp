This folder contains a translated from Python to C++ code for openCV.
For some reasons the C++ code performs worse than Python code (not included).
However, the solution to it was to have a CUDA acceleration (https://opencv.org/platforms/cuda/).
The installation for CUDA dev kit - https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
And a tutorial if needed:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
or
https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/

I got stuck with having raising needed flags. So I had go with trial and error. The folder containing CMake flags has two groups: the first, larger one, might be close to being finished, but I haven't tested it; the second, smaller was just an alternative, really, and it didn't work, but I didn't want to delete it just in case it could prove usefull somehow.
