# implementation for a population in GA
# author: Nabhonil Kar (nkar@princeton.edu)
import Genotype
import Chromosome
import numpy as np


class Member():
    def __init__(self, chromo=None, data_set = None, status=None, acc=None):
        # if data_set is not specified
        if data_set is None:
            raise Exception("Invalid member arguments: data_set argument must be specified.)")
        # is only data_set is provided argument is provided
        elif chromo is None and data_set is not None and status is None and acc is None:
            # sample valid parameter settings
            self.data_set = data_set
            str_len = 8
            self.status = False
            self.acc = 0

            while True:
                rand_int_vals = np.random.randint(1, 2 ** str_len, size=6)
                if self.is_valid_params(rand_int_vals, data_set): break

            self.chromo = Chromosome.Chromo(Genotype.Gene(int(rand_int_vals[0])), Genotype.Gene(int(rand_int_vals[1])),
                                            Genotype.Gene(int(rand_int_vals[2])), Genotype.Gene(int(rand_int_vals[3])),
                                            Genotype.Gene(int(rand_int_vals[4])), Genotype.Gene(int(rand_int_vals[5])))
        # if all arguments are provided
        elif chromo is not None and data_set is not None and status is not None and acc is not None:
            assert type(chromo) is Chromosome.Chromo, 'chromo argument not of type Chromosome.Chromo'
            assert type(data_set) is str, 'data_set argument not of type str'
            assert type(status) is bool, 'status argument not of type bool'
            param_vals = chromo.get_params()
            assert self.is_valid_params(param_vals, data_set),\
                'Invalid chromosome parameters for the given data_set (%s).' % data_set
            assert 0 <= acc <= 100, 'acc argument not in range [0,100]'
            self.chromo = chromo
            self.data_set = data_set
            self.status = status
            self.acc = acc
        # if some arguments are provided
        else:
            raise Exception("Invalid member input: specific all three arguments or none.")

    # check if the chromo parameters are valid parameter settings for data_set
    def is_valid_params(self, param_vals, data_set):
        input_dims = {'MNIST': 28, 'CIFAR10': 32, 'CIFAR100': 32}
        input_dim = input_dims[data_set]
        conv1win_dim = int(param_vals[0])
        conv2win_dim = int(param_vals[2])
        linear1_dim = int(param_vals[4])
        linear2_dim = int(param_vals[5])

        # if first window is larger than input volume dimension, return False
        if conv1win_dim > input_dim: return False
        # apply first convolution
        post_conv1_dim = input_dim - conv1win_dim + 1
        # if post first convolution dimension is odd, return False
        if post_conv1_dim % 2 != 0: return False
        # apply max-pooling
        post_max_pool1_dim = int(post_conv1_dim / 2)
        # if second window is larger than input volume dimension, return False
        if conv2win_dim > post_max_pool1_dim: return False
        # apply second convolution
        post_conv2_dim = post_max_pool1_dim - conv2win_dim + 1
        # if post second convolution dimension is odd, return False
        if post_conv2_dim % 2 != 0: return False

        # check that first linear layer is larger than second
        if linear1_dim <= linear2_dim: return False

        return True

    # returns object's chromo
    def get_chromo(self):
        return self.chromo

    # returns object's data_set
    def get_data_set(self):
        return self.data_set

    # returns object's status
    def get_status(self):
        return self.status

    # sets object's status in place
    def set_status_(self, status):
        assert type(status) is bool, 'status argument not of type bool'
        self.status = status

    # returns object's acc
    def get_acc(self):
        return self.acc

    # set object's acc in place
    def set_acc_(self, acc):
        assert 0 <= acc <= 100, 'acc argument not in range [0,100]'
        self.acc = acc

    # string representation
    def __repr__(self):
        return "%s\n%sstatus: %s; \tacc: %.1f%%\n" % (self.data_set, self.chromo.__repr__(), self.status, self.acc)


class Pop():
    # construct a population of n random members
    def __init__(self, n=0, data_set = None):
        if data_set is None:
            raise Exception('Invalid Pop arguments: data_set argument must be specified')

        self.n = n
        self.data_set = data_set
        self.member_list = [Member(data_set=data_set) for i in range(n)]

    # return list of members
    def to_list(self):
        return self.member_list

    # returns number of members
    def get_n(self):
        return self.n

    # adds the member mem to the population in place
    def add_(self, mem):
        assert type(mem) is Member, 'mem argument is not Member'
        assert mem.get_data_set() is self.data_set, 'Invalid mem data_set: adding %s Member to %s Pop' % (mem.get_data_set(), self.data_set)
        self.n += 1
        self.member_list.append((mem))

    # create the offpsring population by sampling two parents repeatedly, crossing over and mutating
    def offspring(self):
        # check is the whole population has been evaluated by the fitness function
        for mem in self.member_list:
            if mem.get_status() is False:
                raise Exception('Invalid offspring call. All population members have not been evaluated.')

        # initialize an empty Population
        offspring_pop = Pop(data_set=self.data_set)

        # get accumulated, normalized fitness scores
        eval_scores_accum, sort_IDX = self.accum()

        # for n times...
        for i in range(self.n):
            # select two random parents depending on fitness
            b1 = np.random.random()
            b2 = np.random.random()
            parent1_IDX = sort_IDX[np.where(b1 < eval_scores_accum)[0][0]]
            parent2_IDX = sort_IDX[np.where(b2 < eval_scores_accum)[0][0]]

            # get the parents' chromosomes
            chromo1 = self.member_list[parent1_IDX].get_chromo()
            chromo2 = self.member_list[parent2_IDX].get_chromo()

            # create a valid offspring child as a Member object
            while True:
                child_chromo = chromo1.crossover(chromo2)
                param_vals = child_chromo.get_params()
                if self.is_valid_params(param_vals, self.data_set): break

            # mutate the child and ensure it is a valid mutation
            while True:
                child_chromo_mut = child_chromo
                child_chromo_mut.mutate_()
                param_vals = child_chromo_mut.get_params()
                if self.is_valid_params(param_vals, self.data_set): break

            # add to the population
            child = Member(chromo=child_chromo_mut, data_set=self.data_set, status=False, acc=0)
            offspring_pop.add_(child)

        return offspring_pop


    # returns the accumulated, normalized fitness scores of an evaluated population
    def accum(self):
        # check is the whole population has been evaluated by the fitness function
        for mem in self.member_list:
            if mem.get_status() is False:
                raise Exception('Invalid accum call. All population members have not been evaluated.')

        # get the fitness evaluation scores and store them
        eval_scores = np.zeros(self.n)
        eval_total = 0
        for i, mem in enumerate(self.member_list):
            acc = mem.get_acc()
            eval_scores[i] = acc
            eval_total += acc

        # normalize the evaluation scores by the total evaluation score
        eval_scores /= eval_total

        # compute the accumulated evaluation scores
        sort_IDX = np.argsort(eval_scores)
        eval_scores_accum = np.zeros(self.n)
        for i in range(self.n):
            eval_scores_accum[i] = np.sum(eval_scores[sort_IDX[:i+1]])

        # return accumulating sums and indices
        return eval_scores_accum, sort_IDX

    # check if the chromo parameters are valid parameter settings for data_set
    def is_valid_params(self, param_vals, data_set):
        input_dims = {'MNIST': 28, 'CIFAR10': 32, 'CIFAR100': 32}
        input_dim = input_dims[data_set]
        conv1win_dim = int(param_vals[0])
        conv2win_dim = int(param_vals[2])
        linear1_dim = int(param_vals[4])
        linear2_dim = int(param_vals[5])

        # if first window is larger than input volume dimension, return False
        if conv1win_dim > input_dim: return False
        # apply first convolution
        post_conv1_dim = input_dim - conv1win_dim + 1
        # if post first convolution dimension is odd, return False
        if post_conv1_dim % 2 != 0: return False
        # apply max-pooling
        post_max_pool1_dim = int(post_conv1_dim / 2)
        # if second window is larger than input volume dimension, return False
        if conv2win_dim > post_max_pool1_dim: return False
        # apply second convolution
        post_conv2_dim = post_max_pool1_dim - conv2win_dim + 1
        # if post second convolution dimension is odd, return False
        if post_conv2_dim % 2 != 0: return False

        # check that first linear layer is larger than second
        if linear1_dim <= linear2_dim: return False

        return True

    # return a list of accuracies for all members in the population
    def get_all_acc(self):
        # check is the whole population has been evaluated by the fitness function
        for mem in self.member_list:
            if mem.get_status() is False:
                raise Exception('Invalid get_mean_acc call. All population members have not been evaluated.')

        accs = []
        for mem in self.member_list:
            accs.append(mem.get_acc())

        return accs

    # return the average accuracy of an evaluated population
    def get_mean_acc(self):
        # check is the whole population has been evaluated by the fitness function
        for mem in self.member_list:
            if mem.get_status() is False:
                raise Exception('Invalid get_mean_acc call. All population members have not been evaluated.')

        sum_acc = sum(self.get_all_acc())

        return sum_acc/self.n

    # string representation
    def __repr__(self):
        s = "%d %s members:\n" % (self.n, self.data_set)
        for i, member in enumerate(self.member_list):
            s += '%d:\n%s' % (i+1, member.__repr__())
        return s

'''
gene1 = Genotype.Gene(5)
gene2 = Genotype.Gene(16)
gene3 = Genotype.Gene(3)
gene4 = Genotype.Gene(4)
gene5 = Genotype.Gene(120)
gene6 = Genotype.Gene(84)
chromo1 = Chromosome.Chromo(gene1, gene2, gene3, gene4, gene5, gene6)

x1 = Member(chromo=chromo1, data_set='MNIST', status=True, acc=99)
#print(x1)

x2 = Member(data_set='MNIST')
x2.set_status_(True)
x2.set_acc_(85)
#print(x2)

p1 = Pop(data_set='MNIST')
#print(p1)

p1.add_(x1)
#print(p1)

p1.add_(x2)
#print(p1)

print(p1)
print(p1.offspring())

p2 = Pop(3, data_set='MNIST')
#print(p2)
'''