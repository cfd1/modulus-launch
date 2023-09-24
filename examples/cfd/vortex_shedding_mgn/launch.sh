python -m pip uninstall nvidia-modulus nvidia-modulus.sym nvidia-modulus.launch -y

cd /modulus/
python -m pip install -e .

cd /modulus-sym/
python -m pip install -e .

cd /modulus-launch/
python -m pip install -e .

cd /modulus-launch/examples/cfd/vortex_shedding_mgn/
git config --global --add safe.directory /modulus-launch

pip install wandb --upgrade

python wandb_train.py "$@"
