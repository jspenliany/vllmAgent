ubuntu 22.04.5
6.8.0.40.generic
rtx 3090
nvidia-driver-580.95.05      CUDA Version: 13.0
gcc 12.3


sudo apt install build-essential cmake git

sudo apt install nvidia-cuda-toolkit

mkdir llama-src
cd llama-src/
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp/


cmake -B build -DGGML_CUDA=ON
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF


wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
sudo sh cuda_13.0.0_580.65.06_linux.run

Toolkit:  Installed in /usr/local/cuda-13.0/
Please make sure that
 -   PATH includes /usr/local/cuda-13.0/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-13.0/lib64, or, add /usr/local/cuda-13.0/lib64 to /etc/ld.so.conf and run ldconfig as root
To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-13.0/bin

nano ~/.bashrc
source ~/.bashrc

cmake --build build --config Release

after build process, there must generated 'libggml-cuda.sXXX'-like files. 
such files will leverage your nvidia GPU device.



