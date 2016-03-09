# PBS -l:ngpus=1,nodes=1
# PBS -M clay.l.mcleod@gmail.com

export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
python src/experiments/mnist/mnist.py mrelu -l 0.01
