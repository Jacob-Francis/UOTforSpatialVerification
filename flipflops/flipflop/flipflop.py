"""
`FlipFlop`: detect when a sequence of numbers starts repeating itself.

Used by `UnbalancedSinkhorn.sinkhorn_algorithm` as an early-stopping check. At each
convergence check the Sinkhorn error ("err", the largest change in the potentials)
is passed to `FlipFlop.oscialltion`. Near floating-point precision the error can stop
decreasing and instead bounce between the same few values (a "flip-flop"). Once
enough repeats are seen, the loop stops instead of iterating forever.

How it works
------------
- Keep the last `length` values in a fixed-size queue (initially all 0.0).
- Each new value that is ALREADY in the queue counts as a repeat (`self.count += 1`).
- A new value that is NOT in the queue resets the count to 0 (with `memory=1`).
- `oscialltion` returns True once `count >= oscialltion_count`, i.e. after that many
  repeats in a row.
- If the whole queue becomes one identical value, convergence has stalled
  completely: warn and return True straight away.

Values are compared with exact equality (`a in queue` uses `==`), so only exact,
bit-for-bit repeats count. That is the intended signal: genuinely stalled iterations
reproduce the same floating-point numbers.

Usage in the Sinkhorn loop: `FlipFlop(6, 3)` for the first run of iterations, then
`FlipFlop(4, 2)` after each restart. Note `FlipFlop(6, 3)` cannot detect a cycle
between just two values; see the BUG note in `counting_oscilliations`.
"""
from collections import deque
import warnings


class FlipFlop:
    """
    Detect repeating values in a stream of numbers (see module docstring).

    Parameters
    ----------
    length : int, optional
        Number of recent values to remember, by default 5.
    oscialltion_count : int, optional
        Number of repeats in a row needed before `oscialltion` returns True,
        by default 2.
    memory : int, optional
        Number of non-repeating values in a row allowed before the repeat count is
        reset, by default 1. Only the default works; see the BUG note in
        `counting_oscilliations`.

    Attributes
    ----------
    queue : collections.deque
        The last `length` values, oldest on the left. Starts as `length` zeros.
    count : int
        Current number of repeats in a row.
    """

    def __init__(self, length=5, oscialltion_count=2, memory=1):
        # length of list to check for flip flopping
        self.length = length
        self.queue = deque([])
        self.count = 0
        self.oscialltion_count = oscialltion_count
        self.memory = memory
        # Pre-fill with zeros so the queue always has exactly `length` entries.
        # NOTE: an error of exactly 0.0 therefore counts as a repeat straight away.
        for k in range(length):
            self.queue.append(0.0)

    def _add(self, a):
        """
        Push `a` onto the queue, dropping the oldest value.

        Parameters
        ----------
        a : float or 0-d torch.Tensor
            New value.

        Returns
        -------
        bool
            True if `a` was already in the queue BEFORE it was added (a repeat),
            False otherwise.
        """
        # check not repeated twice within some tol otherwise add one and remove one
        # using this won't increase the oscialtion count
        # (Despite the comment above, the check is exact equality, not a tolerance.)
        if a in self.queue:
            self.queue.popleft()
            self.queue.append(a)
            return True
        else:
            self.queue.popleft()
            self.queue.append(a)
            return False

    def counting_oscilliations(self, a):
        """
        Add `a` to the queue and update the repeat count.

        - Repeat, and `a` now appears at most twice in the queue: count += 1.
        - Repeat, but `a` already appears 3+ times: count unchanged. A long run of
          one value is handled by the "completely plateaued" check instead.
        - Whole queue equal to `a`: warn, and set count straight to the threshold.
        - Not a repeat: reset count to 0 (once `memory` non-repeats are reached).

        Parameters
        ----------
        a : float or 0-d torch.Tensor
            New value.
        """
        # could have a count of how many times there's a repeated 'add' then provided theres two i.e some kind of oscialltion stop? memeory?
        # BUG(stage 4): memory_count is a local variable, reset to 0 on every call,
        # so it can never get past 1. With memory > 1 the count therefore never
        # resets (checked: FlipFlop(memory=2) kept count=1 after 4 non-repeats).
        # It needs to be stored on self to persist between calls.
        memory_count = 0
        if self._add(a):
            # BUG(stage 4): this "3+ copies" rule caps the count for a 2-value cycle
            # (a, b, a, b, ...) at 2, because each value soon appears 3 times in the
            # queue. So FlipFlop(6, 3), which the solver uses first, never detects a
            # 2-value flip-flop (checked over 40 steps); it does detect 3-value
            # cycles. FlipFlop(4, 2), used after a restart, detects both.
            if self.queue.count(a) > 2:  # i.e. theres three of the same number here
                pass
            else:
                self.count += 1
            memory_count = 0

            if self.queue.count(a) == self.length:
                warnings.warn(
                    "Convergence has completely plateaued"
                )  # if all same in list
                # Force `oscialltion` to return True on this call.
                self.count = self.oscialltion_count
        else:
            memory_count += 1
            if memory_count >= self.memory:
                self.count = 0

    def oscialltion(self, a):
        """
        Add `a` and report whether the sequence is now oscillating.

        This is the only method the Sinkhorn loop calls. ("oscialltion" is a typo for
        "oscillation"; the name is kept for now because the solver calls it.)

        Parameters
        ----------
        a : float or 0-d torch.Tensor
            New value, e.g. the latest Sinkhorn error.

        Returns
        -------
        bool
            True once there have been `oscialltion_count` repeats in a row, or the
            whole queue is one value.
        """
        # counting for oscialltion_count amount of repeats, could be the same number twice and had to be subsequent
        self.counting_oscilliations(a)

        if self.count >= self.oscialltion_count:
            return True
        else:
            return False


# to be made into tests
# (Stage 3: turn the commented-out checks below into a real test file.)
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
