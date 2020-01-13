# implementation for a genotype for a GA
# author: Nabhonil Kar (nkar@princeton.edu)
import numpy as np


class Gene():
    # instantiation
    def __init__(self, int_val):
        assert type(int_val) is int, 'argument int_val is not integer'
        if int_val >= 2**8: raise Exception('Invalid chromo input: value is 256 or greater.')
        elif int_val <= 0: raise Exception('Invalid chromo input: value is 0 or less.')
        self.int_val = int_val
        self.bin_val = self.to_bin(int_val)

    # return integer value
    def get_int(self):
        return self.int_val

    # return string binary
    def get_bin(self):
        return self.bin_val

    # converts int to binary
    def to_bin(self, int_val):
        assert type(int_val) is int, 'argument int_val is not integer'
        bin_val = bin(int_val)[2:]
        # append additional 0's if necessary (desired string length of 8)
        str_len = 8
        bin_val_len = len(bin_val)
        if bin_val_len < str_len:
            bin_val = '0'*(str_len-bin_val_len) + bin_val
        return bin_val

    # converts binary to int
    def to_int(self, bin_val):
        assert type(bin_val) is str, 'argument bin_val is not str'
        return int(bin_val, 2)

    # crossover self with Gene2
    def crossover(self, Gene2, true_cross = True):
        assert type(Gene2) is Gene, 'argument Gene2 is not Genotype.Gene'
        assert type(true_cross) is bool, 'argument true_cross is not bool'
        # get two random positions on the ring chromosome
        str_len = 8
        pos1 = np.random.randint(0, str_len)
        pos2 = np.random.randint(0, str_len)
        # if true_cross == True, then we are guaranteed some genes from each parent
        # if false, we may have an offspring that is identical to a parent
        # (though this is also possible if true_cross == True)
        if true_cross == True:
            while pos1 == pos2: pos2 = np.random.randint(0, str_len)

        # take segments from self and chromo2 to construct an offspring's binary and int values
        if pos2 > pos1:
            seg1 = self.bin_val[:pos1]
            seg2 = Gene2.bin_val[pos1:pos2]
            seg3 = self.bin_val[pos2:]
        else:
            seg1 = self.bin_val[:pos2]
            seg2 = Gene2.bin_val[pos2:pos1]
            seg3 = self.bin_val[pos1:]
        bin_val = seg1 + seg2 + seg3
        int_val = self.to_int(bin_val)

        # check to avoid the case that the value is 0
        if int_val == 0:
            return self.crossover(Gene2)
        else:
            return Gene(int_val)

    # does an in place mutation each gene of the binary representation w.p. p
    def mutate_(self, p = 0.001):
        if p < 0 or p > 1: raise Exception('Invalid mutate_ inpute: probability p is outside of range [0,1].')
        # get a random binary mask of length str_len with each bit as 1 w.p. p
        str_len = 8
        mask_bin_val = np.array_str(np.random.binomial(1, p, str_len))[1::2]
        new_int_val = self.to_int(mask_bin_val) ^ self.int_val
        # check to avoid the case that the value is 0
        if new_int_val == 0:
            self.mutate_(p = p)
        else:
            self.bin_val = self.to_bin(new_int_val)
            self.int_val = new_int_val

    # string representation
    def __repr__(self):
        return "%3d : %s" % (self.int_val, self.bin_val)

'''
x1 = Gene(255)
x2 = Gene(1)
print(x1)
print(x2)
x3 = x1.crossover(x2)
print(x3)
x3.mutate_(p=0.2)
print(x3)
'''