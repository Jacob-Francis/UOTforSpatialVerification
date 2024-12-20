from collections import deque
import warnings


class FlipFlop:
    def __init__(self, length=5, oscialltion_count=2, memory=1):
        # length of list to check for flip flopping
        self.length = length
        self.queue = deque([])
        self.count = 0
        self.oscialltion_count = oscialltion_count
        self.memory = memory
        for k in range(length):
            self.queue.append(0.0)

    def _add(self, a):
        # check not repeated twice within some tol otherwise add one and remove one
        # using this won't increase the oscialtion count
        if a in self.queue:
            self.queue.popleft()
            self.queue.append(a)
            return True
        else:
            self.queue.popleft()
            self.queue.append(a)
            return False

    def counting_oscilliations(self, a):
        # could have a count of how many times there's a repeated 'add' then provided theres two i.e some kind of oscialltion stop? memeory?
        memory_count = 0
        if self._add(a):
            if self.queue.count(a) > 2:  # i.e. theres three of the same number here
                pass
            else:
                self.count += 1
            memory_count = 0

            if self.queue.count(a) == self.length:
                warnings.warn(
                    "Convergence has completely plateaued"
                )  # if all same in list
                self.count = self.oscialltion_count
        else:
            memory_count += 1
            if memory_count >= self.memory:
                self.count = 0

    def oscialltion(self, a):
        # counting for oscialltion_count amount of repeats, could be the same number twice and had to be subsequent
        self.counting_oscilliations(a)

        if self.count >= self.oscialltion_count:
            return True
        else:
            return False


# to be made into tests
if __name__ == "__main__":
    pass
    # testing osciallting adding and finding repeats correctly
    # ff = FlipFlop(4)
    # print(ff.queue)
    # import torch
    # for k in range(7):
    #     x = torch.rand(1)
    #     ff.oscialltion(x)
    #     print(x)
    #     xo = x.clone()
    # print(ff.oscialltion(xo))
    # print(ff.queue)
    # print(ff.oscialltion(ff.queue[1])==True)
    # print(ff.queue)

    # del ff
    ## testing osciallting flags false correctly
    # ff = FlipFlop(4)
    # print(ff.queue)
    # import torch
    # for k in range(7):
    #     x = torch.rand(1)
    #     ff.oscialltion(x)
    #     print(x)
    #     xo = x.clone()
    # print(ff.oscialltion(xo))
    # print(ff.oscialltion(torch.rand(1)))
    # print(ff.queue)
    # print(ff.oscialltion(ff.queue[2])==False)
    # print(ff.queue)

    # # testing osciallting flags True correctly with longer memory
    # ff = FlipFlop(4, memory=2)
    # print(ff.queue)
    # import torch
    # for k in range(7):
    #     x = torch.rand(1)
    #     ff.oscialltion(x)
    #     print(x)
    #     xo = x.clone()
    # print(ff.oscialltion(xo))
    # print(ff.oscialltion(torch.rand(1)))
    # print(ff.queue)
    # print(ff.oscialltion(ff.queue[2])==True)
    # print(ff.queue)

    # # testing osciallting flags True if all equal
    # ff = FlipFlop(4, memory=1)
    # print(ff.queue)
    # import torch
    # for k in range(7):
    #     x = torch.rand(1)
    #     ff.oscialltion(x)
    #     xo = x.clone()
    # print('created list')
    # for _ in range(2):
    #     print(ff.oscialltion(xo))
    # print(ff.oscialltion(xo))
    # print(ff.queue)
