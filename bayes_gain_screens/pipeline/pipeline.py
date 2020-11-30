import os
import datetime
from timeit import default_timer
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

def str_(s):
    """
    In python 3 turns bytes to string. In python 2 just passes string.
    :param s:
    :return:
    """
    try:
        return s.decode()
    except:
        return s


def iterative_topological_sort(graph, start):
    """
    Get Depth-first topology.

    :param graph: dependency dict (like a dask)
        {'a':['b','c'],
        'c':['b'],
        'b':[]}
    :param start: str
        the node you want to search from.
        This is equivalent to the node you want to compute.
    :return: list of str
        The order get from `start` to all ancestors in DFS.
    """
    seen = set()
    stack = []  # path variable is gone, stack and order are new
    order = []  # order will be in reverse order at first
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v)  # no need to append to path any more
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:  # new stuff here!
                order.append(stack.pop())
            stack.append(v)
    return stack + order[::-1]  # new return value!


def now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def execute_dask(dsk, key, timing_file=None, state_file=None, retry_task_on_fail=0):
    """
    Go through the dask in topo order using DFS to reach `key`
    :param dsk: dict
        Dask graph
    :param key: str
        The node you want to arrive at.
    :param timing_file: str
        Where to store timing info
    :param state_file:
        Where to store pipeline state

    Args:
        dsk: Dask graph of Step.get_dask_task
        key: The node you want to arrive at.
        timing_file: Where to store timing info
        state_file: Where to store pipeline state
        retry_task_on_fail: int, how many times to retry if a failure.

    Returns: dictionary of exit codes.
    """
    graph = {k: v[1:] for k, v in dsk.items()}
    topo_order = iterative_topological_sort(graph, key)[::-1]
    logger.info("Execution order shall be:")
    for k in topo_order:
        logger.info("\t{}".format(k))
    res = {}
    with open(state_file, 'w') as state:
        state.write("{} | START_PIPELINE\n".format(now()))
        state.flush()
        for k in topo_order:
            logger.info("{} | Executing task {}".format(now(), k))
            state.write("{} | START | {}\n".format(now(), k))
            state.flush()
            try_idx = 0
            while True:
                t0 = default_timer()
                res[k] = dsk[k][0]()
                time_to_run = default_timer() - t0
                if res[k] is not None:
                    logger.info("Task {} took {:.2f} hours".format(k, time_to_run / 3600.))
                    if res[k] == 0:
                        state.write("{} | END | {}\n".format(now(), k))
                        state.flush()
                        if timing_file is not None:
                            update_timing(timing_file, k, time_to_run)
                        break
                    else:
                        if try_idx <= retry_task_on_fail:
                            try_idx += 1
                            state.write("{} | RETRY | {}\n".format(now(), k))
                            continue
                        state.write("{} | FAIL | {}\n".format(now(), k))
                        state.flush()
                        logger.info("FAILURE at: {}".format(k))
                        state.write("{} | PIPELINE_FAILURE\n".format(now()))
                        state.flush()
                        exit(3)
                else:
                    state.write("{} | END_WITHOUT_RUN | {}\n".format(now(), k))
                    logger.info("{} skipped.".format(k))
                    break
        state.write("{} | PIPELINE_SUCCESS\n".format(now()))
        state.flush()
    return res


def update_timing(timing_file, name, time):
    logger.info("Updating timing file.")
    timings = OrderedDict()
    if os.path.isfile(timing_file):
        with open(timing_file, 'r+') as f:
            for line in f:
                if line.strip() == "" or "#" in line:
                    continue
                line = [a.strip() for a in line.strip().split(',')]
                timings[line[0]] = line[1:]
    if name not in timings.keys():
        timings[name] = ["{:.2f}".format(time)]
    else:
        timings[name].append("{:.2f}".format(time))
    with open(timing_file, 'w') as f:
        for k, t in timings.items():
            f.write("{},{}\n".format(k, ",".join(t)))


def setup_auto_resume(auto_resume, state_file, steps):
    force_resume = False
    if auto_resume < 0:
        force_resume = True
        auto_resume = - auto_resume

    if auto_resume:
        logger.info("Attempting auto resume")
        if not os.path.isfile(state_file):
            logger.info("No state file: {}".format(state_file))
            logger.info("Resume not possible. Trusting your user requested pipeline steps.")
        else:
            if not force_resume:
                with open(state_file, 'r') as f:
                    applicable = False
                    for line in f.readlines():
                        if "PIPELINE_FAILURE" in line or "PIPELINE_SUCCESS" in line:
                            applicable = True
                if not applicable:
                    raise ValueError("The previous run did not finish, but trying to do auto-resume.")
            if auto_resume == 1:
                logger.info("Resuming pipeline with flag setting '1'. Deleting old undone/failed work.")
            if auto_resume == 2:
                logger.info("Resuming pipeline with flag setting '2'. Co-existing with old undone/failed work.")
            for step in steps.keys():
                if steps[step].flag is None:
                    logger.info("Step {} being skipped.".format(step))
                    steps[step].flag = 0
                if steps[step].flag > 0:
                    logger.info(
                        "Changing step user requested flag {} : {} -> {}".format(step, steps[step].flag, auto_resume))
                    steps[step].flag = auto_resume
            with open(state_file, 'r') as f:
                for line in f.readlines():
                    if "END" not in line:
                        continue
                    if step == 'endpoint':
                        continue
                    step = line.split(" ")[-1].strip()
                    if step not in steps.keys():
                        raise ValueError("Could not find step {}".format(step))
                    logger.info("Auto-resume infers {} should be skipped.".format(step))
                    if steps[step].flag > 0:
                        logger.info(
                            "Changing step user requested flag {} : {} -> {}".format(step, steps[step].flag,
                                                                                     0))
                        steps[step].flag = 0


class Pipeline(object):
    def __init__(self, auto_resume, root_working_dir, state_file, timing_file, steps):
        self._steps = steps
        # possibly auto resuming by setting flag
        setup_auto_resume(self._auto_resume, self._state_file, self._steps)
        self._root_working_dir = root_working_dir
        for k, step in self._steps.items():
            step.build_working_dir(self._root_working_dir)
        self._state_file = state_file
        self._timing_file = timing_file
        self._auto_resume = auto_resume

    def build(self):
        """
        Builds the commands required to run the pipeline, and sets autoresume flags.
        """
        logger.info("Building the pipeline.")
        # make required working directories (no deleting
        for k, step in self._steps.items():
            step.build_cmd()
            step.cmd.add('working_dir', step.working_dir)

    def run(self, retry_task_on_fail=0):
        """
        Run the pipieline in topo order, retrying a failed task if desired.

        Args:
            retry_task_on_fail: How many times to retry on fail. Useful when the compute resources are
                non-deterministically buggy.
        """
        logger.info("Running the pipeline.")
        dsk = {}
        for name in self._steps.keys():
            dsk[name] = self._steps[name].get_dask_task()

        dsk['endpoint'] = (lambda *x: None,) + tuple([k for k in dsk.keys()])
        execute_dask(dsk, 'endpoint', timing_file=self._timing_file, state_file=self._state_file,
                     retry_task_on_fail=retry_task_on_fail)



