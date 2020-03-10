#!/bin/bash

sudo apt update
sudo apt upgrade -y
sudo apt install -y git build-essential libbz2-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev tk-dev libpng-dev libfreetype6-dev python-pip
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
echo -e '\n' >> ~/.bashrc
echo -e 'export PATH="~/.pyenv/bin:$PATH"' >> ~/.bashrc
echo -e 'eval "$(pyenv init -)"' >> ~/.bashrc
echo -e 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
echo 'CLOSE THIS TERMINAL AND OPEN A NEW ONE'