apt-get update && apt-get install curl zip unzip git fish ssh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh 
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz
sudo rm -rf /opt/nvim-linux-x86_64 
sudo tar -C /opt -xzf nvim-linux-x86_64.tar.gz

