# PBS -l:ngpus=1,nodes=1
# PBS -M clay.l.mcleod@gmail.com

export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
python src/deepcnet.py relu small -n -d 0.2
