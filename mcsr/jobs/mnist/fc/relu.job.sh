# PBS -l:ngpus=1,nodes=1
# PBS -M clay.l.mcleod@gmail.com
# PBS -N relu-mfc

export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
python src/mnist_fc.py relu -l ${lr}
