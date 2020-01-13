# implementation for a chromosome in GA
# author: Nabhonil Kar (nkar@princeton.edu)
import numpy as np
import Genotype


class Chromo():
    def __init__(self, conv1win, conv1chan, conv2win, conv2chan, linear1, linear2):
        assert type(conv1win) is Genotype.Gene, 'argument conv1win is not Genotype.Gene'
        assert type(conv1chan) is Genotype.Gene, 'argument conv1chan is not Genotype.Gene'
        assert type(conv2win) is Genotype.Gene, 'argument conv2win is not Genotype.Gene'
        assert type(conv2chan) is Genotype.Gene, 'argument conv2chan is not Genotype.Gene'
        assert type(linear1) is Genotype.Gene, 'argument linear1 is not Genotype.Gene'
        assert type(linear2) is Genotype.Gene, 'argument linear2 is not Genotype.Gene'
        self.conv1win = conv1win
        self.conv1chan = conv1chan
        self.conv2win = conv2win
        self.conv2chan = conv2chan
        self.linear1 = linear1
        self.linear2 = linear2

    # performs crossover between the genotypes of self and Chromo2
    def crossover(self, Chromo2):
        assert type(Chromo2) is Chromo, 'argument Chromo2 is not Chromo'
        conv1win_new = self.conv1win.crossover(Chromo2.conv1win)
        conv1chan_new = self.conv1chan.crossover(Chromo2.conv1chan)
        conv2win_new = self.conv2win.crossover(Chromo2.conv2win)
        conv2chan_new = self.conv2chan.crossover(Chromo2.conv2chan)
        linear1_new = self.linear1.crossover(Chromo2.linear1)
        linear2_new = self.linear2.crossover(Chromo2.linear2)
        return Chromo(conv1win_new, conv1chan_new, conv2win_new, conv2chan_new, linear1_new, linear2_new)

    # mutate all genes within the chromosome w.p. p
    def mutate_(self, p=0.001):
        if p < 0 or p > 1: raise Exception('Invalid mutate_ input: probability p is outside of range [0,1].')
        self.conv1win.mutate_(p=p)
        self.conv1chan.mutate_(p=p)
        self.conv2win.mutate_(p=p)
        self.conv2chan.mutate_(p=p)
        self.linear1.mutate_(p=p)
        self.linear2.mutate_(p=p)

    # return an numpy array of all parameter values
    def get_params(self):
        return np.array([int(self.get_conv1win_int()), int(self.get_conv1chan_int()),
                  int(self.get_conv2win_int()), int(self.get_conv2chan_int()),
                  int(self.get_linear1_int()), int(self.get_linear2_int())])

    # return the int representation of the first convolutional layer window size
    def get_conv1win_int(self):
        return self.conv1win.get_int()

    # return the int representation of the first convolutional layer window size
    def get_conv1chan_int(self):
        return self.conv1chan.get_int()

    # return the int representation of the second convolutional layer window size
    def get_conv2win_int(self):
        return self.conv2win.get_int()

    # return the int representation of the second convolutional layer window size
    def get_conv2chan_int(self):
        return self.conv2chan.get_int()

    # return the int representation of the first linear gene
    def get_linear1_int(self):
        return self.linear1.get_int()

    # return the binary representation of the first linear gene
    def get_linear1_bin(self):
        return self.linear1.get_bin()

    # return the int representation of the second linear gene
    def get_linear2_int(self):
        return self.linear2.get_int()

    # return the binary representation of the second linear gene
    def get_linear2_bin(self):
        return self.linear2.get_bin()

    # string representation
    def __repr__(self):
        return 'Conv1win:\t%s; Conv1chan:\t%s\nConv2win:\t%s; Conv2chan:\t%s\nLinear1:\t%s; Linear2:\t%s\n' % \
               (self.conv1win.__repr__(), self.conv1chan.__repr__(),
                self.conv2win.__repr__(), self.conv2chan.__repr__(),
                self.linear1.__repr__(), self.linear2.__repr__())

'''
x1 = Chromo(Genotype.Gene(5),Genotype.Gene(16),Genotype.Gene(4),Genotype.Gene(4),Genotype.Gene(120),Genotype.Gene(84))
x2 = Chromo(Genotype.Gene(5),Genotype.Gene(16),Genotype.Gene(4),Genotype.Gene(4),Genotype.Gene(81),Genotype.Gene(61))

x3 = x1.crossover(x2)
x3.mutate_(p=0.01)
print(x3)
print(x3.get_linear1_int(), x3.get_linear2_int())
print(x3.get_linear1_bin(), x3.get_linear2_bin())
'''