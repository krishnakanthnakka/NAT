
#conda activate tap
DIR="/workspace/NAT"
echo $DIR
export PYTHONPATH=$DIR/pix2pix:$PYTHONPATH
export PYTHONPATH=$DIR/CDA/cda:$PYTHONPATH
export PYTHONPATH=$DIR/cda/pipeline:$PYTHONPATH
export PYTHONPATH=$DIR/cda/pipeline/sparse_autoencoder:$PYTHONPATH

pip install ray==2.9.2
# pip install pyarrow==14.0.2
# pip install pandas==2.1.4
# pip install scikit-learn==1.3.0
#pip install timm #==1.0.9
# pip install opencv-python==4.10.0.84 ipython==8.18.1
# pip install pretrainedmodels==0.7.4
# pip install setuptools==68.2.2

