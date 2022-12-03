#step1 create virtual environment
conda create -n Resnet18 pytorch matplotlib numpy \
scikit-learn torchvision pandas requests python=3.8
conda activate Resnet18
conda install torchmetrics==0.10.3 -c conda-forge
#Step2 download dataset
git clone https://github.com/rois-codh/kmnist.git
cd kmnist
python download_data.py
##choose to download .npz sformat file
## type in 1 -> 2
#step3 train model
python train.py

#step4 test model
##Test accuracy: 95.97%
acc:  tensor(95.9700)
confuse matrix
tensor([[971,   3,   0,   0,  11,   5,   0,   4,   4,   2],
        [  2, 921,  27,   0,   2,   3,  24,   4,   7,  10],
        [ 13,   0, 955,  12,   2,   4,   6,   2,   2,   4],
        [  0,   0,   5, 976,   0,  11,   4,   2,   2,   0],
        [ 28,   0,   1,   3, 932,   7,   3,   4,  16,   6],
        [  1,   1,  23,   3,   2, 946,   2,   5,   4,  13],
        [  4,   1,  11,   3,   0,   7, 971,   2,   0,   1],
        [  6,   0,   0,   2,   4,   1,   2, 971,   7,   7],
        [  0,   4,  12,   2,   1,   2,   0,   1, 978,   0],
        [  3,   1,   5,   2,   1,   3,   6,   1,   2, 976]])