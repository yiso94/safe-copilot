sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y wget python3-dev

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda-repo-ubuntu2204-13-1-local_13.1.1-590.48.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-13-1-local_13.1.1-590.48.01-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:$CPATH' >> ~/.bashrc

curl -LsSf https://astral.sh/uv/install.sh | sh

echo 'export OPENAI_API_KEY=ollama' >> ~/.bashrc
echo 'export OPENAI_BASE_URL="http://localhost:11434/v1"' >> ~/.bashrc
echo 'export OPENAI_MODEL="qwen3-coder-next:latest"' >> ~/.bashrc

# Install Qwen
bash -c "$(curl -fsSL https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen.sh)"

# Start Ollama server
ollama serve
ollama pull qwen3-coder-next:latest