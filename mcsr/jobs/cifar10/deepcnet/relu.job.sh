# PBS -l:ngpus=1,nodes=1
# PBS -M clay.l.mcleod@gmail.com

rm -rf /home/ums/r1842/.theano/compiledir_Linux-3.0--default-x86_64-with-SuSE-11-x86_64-x86_64-2.7.11-64/lock_dir

export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
python src/deepcnet.py relu reg -l 0.001
