import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_fgsm(fh,look_str):
    vals = []

    file_text = fh.read().replace(',','\n')

    for line in file_text.split('\n'):
        if look_str in line:
            print line
            val = line.split(' ')[-1]
            val = val.replace('%','')
            val = val.replace('accuracy=','')
            val = val.replace('loss=','')
            val = float(val)
            vals.append(val)

    vals = np.array(vals)

    vals = moving_average(vals)

    return vals

if False: 

    fh1 = open('slurm-169386.out', "r")
    fh2 = open('slurm-169387.out', "r")
    fh3 = open('slurm-169388.out', "r")

    look_str = "Test: loss"

    plt.plot(get_fgsm(fh1,look_str))
    plt.plot(get_fgsm(fh2,look_str))
    plt.plot(get_fgsm(fh3,look_str))

    plt.ylim(0.0, 0.9)

    plt.legend(["No Mixup", "Mixup Visible", "Mixup Hidden"],fontsize=16)

elif False: 

    fh1 = open('slurm-171861.out', "r")
    fh2 = open('slurm-171862.out', "r")
    fh3 = open('slurm-171863.out', "r")
    fh4 = open('slurm-172105.out', 'r')
    look_str = "Test with fgsm"
    plt.plot(get_fgsm(fh1,look_str))
    plt.plot(get_fgsm(fh2,look_str))
    plt.plot(get_fgsm(fh3,look_str))
    plt.plot(get_fgsm(fh4,look_str))

elif True:
    fh1 = open('slurm-172345.out', "r")
    fh2 = open('slurm-172355.out', "r")
    fh3 = open('slurm-172361.out', "r")
    fh4 = open('slurm-172363.out', 'r')
    fh5 = open('slurm-172364.out', 'r')
    look_str = "Test with fgsm 0.03"
    plt.plot(get_fgsm(fh1,look_str))
    plt.plot(get_fgsm(fh2,look_str))
    plt.plot(get_fgsm(fh3,look_str))
    plt.plot(get_fgsm(fh4,look_str))
    plt.plot(get_fgsm(fh5,look_str))

else: 

    fh1 = open('slurm-171861.out', "r")
    fh2 = open('slurm-171862.out', "r")
    fh3 = open('slurm-171863.out', "r")

    look_str = "Test: loss"

    plt.plot(get_fgsm(fh1,look_str))
    plt.plot(get_fgsm(fh2,look_str))
    plt.plot(get_fgsm(fh3,look_str))

    plt.ylim(0.0, 0.9)

    plt.legend(["No Mixup", "Mixup Visible", "Mixup Hidden"],fontsize=16)

plt.savefig('scripts/plots.png')


