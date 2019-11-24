# Installation Instruction #

(this file is modified based on `README.md` in https://github.com/gtrll/rlfamily_cv)



Tested in Ubuntu 16.04 and Ubuntu 18.04 with python 3.

#### Install repo and most of the requirements ####
Prepare python3 virtual environment:
```
sudo apt-get install python3-pip
sudo pip3 install virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
```
Install this repo and requirements:
```
git clone https://github.com/Leonardo-H/DR-PG.git
pip install --upgrade -r requirements.txt
```
You may need to run
```
export PYTHONPATH="{PYTHONPATH}:[the parent folder of this repo]"
```

#### Install Dart ####
The Ubuntu package is too new for PyDart2, so we install it manually. 

First install the requirements following the instructions of Install DART from source at https://dartsim.github.io/install_dart_on_ubuntu.html. We compile and install it manually, because PyDart2 only supports Dart before 6.8.
```
git clone git://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.7.2
mkdir build
cd build
cmake ..
make -j4
sudo make install
```
Someitmes you may need to link library manually.
```
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib" >> ~/.bashrc
```

#### Install PyDart2 ####
Installing PyDart2 through pip does not work, so we install it manually.
```
git clone https://github.com/sehoonha/pydart2.git
cd pydart2
python setup.py build build_ext
python setup.py develop
```


#### Install DartEnv ####
This is a slightly modified version of [DartEnv](https://github.com/DartEnv/dart-env). The changes include:

* Make nodisplay as default.
* Add a state property for convenience.
* Added arguments for physical parameter perturbation. 

To install it, 
```
git clone https://github.com/gtrll/dartenv.git
cd dartenv
git checkout cv
pip install -e .[dart]
```

