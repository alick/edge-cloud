#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulation of Edge Cloud Migration

# defaultdict provides default values for missing keys
from collections import defaultdict, deque, OrderedDict
import argparse
import logging
import sys
import math
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path as path


class EdgeCloud():
    """The EdgeCloud class holds all the magic."""

    def __init__(self, path_to_file, K=5, N=None, special=False, M=5):
        """Initialize with a file containing the sequence of requests.

        :param path_to_file: string to specify the path to the file
        :param K: number of servers hosted by edge cloud (default: 5)
        :param M: cost ratio of migration over forwarding (default: 5)
        :param N: only use the first N requests from
        file. Useful for debugging. Default is to use all requests.
        """
        if K >= 1:
            self.K = int(K)
        else:
            raise Exception('The parameter K should be a positive integer.')
        if special and M < 1:
                raise Exception('The parameter M should be at least 1.')
        if N is None:
            self.N = None
        elif N >= 1:
            self.N = int(N)
        else:
            raise Exception('The parameter N should be '
                            'a positive integer or None.')
        self.requests = []
        with open(path_to_file, 'r') as f:
            if self.N is None:
                for line in f:
                    r = int(line)  # one request specified by its service id
                    self.requests.append(r)
                self.N = len(self.requests)
            else:
                for x in range(self.N):
                    line = next(f)
                    r = int(line)
                    self.requests.append(r)

        requests_cnt = defaultdict(int)  # a dict with default integer value 0
        for r in self.requests:
            requests_cnt[r] += 1
        # Number of unique services requested.
        self.N_unique = len(requests_cnt)

        # The requests sorted by values and then keys.
        self.sorted_requests_cnt = sorted(
            requests_cnt.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True)

        # The requests sorted by keys (IDs) in ascending order.
        self.sorted_requests = sorted(requests_cnt.keys())

        # Set of all possible services.
        self.services = set(self.sorted_requests)

        self.gen_het_services(special=special, M=M)

        if self.N_unique <= self.K:
            logging.warning('WARN: Storage can hold all possible '
                            'services!')
            self.K = self.N_unique

        logging.info('No. requests: N={0}'.format(self.N))
        logging.info('No. unique services: |S|={0}'
                     .format(self.N_unique))
        logging.info('Edge cloud capacity: K={0}'.format(self.K))

        # Python 3.5 offers math.inf, which is equivalent to float('inf').
        try:
            math.inf
        except AttributeError:
            math.inf = float('inf')

        self.reset()
        logging.info('Initial edge services: {}'.format(self.edge_services))

    def gen_het_services(self, special=False, M=5):
        self.M = dict()
        self.F = dict()
        self.W = dict()  # space requirement per service
        if not special:
            # Generate heterogeneous service requirements by service ID.
            for s in self.services:
                s = int(s)
                self.F[s] = s % 3 + 1
                # // is floor division (integer division) in Python.
                self.M[s] = ((s // 3) % 3 + 1) * 5
                self.W[s] = (s // 9) % 3 + 1
        else:
            # Reduce to homogeneous case.
            for s in self.services:
                self.M[s] = M
                self.F[s] = 1
                self.W[s] = 1

    def init_edge_services(self):
        """Initialize the services in the edge cloud storage.

        We use the services with lowest ids that can fit in the storage.
        """
        self.K_rem = self.K
        self.edge_services = set()
        for s in self.sorted_requests:
            W = self.W[s]
            # Skip the services which exceed the capacity.
            if W <= self.K_rem:
                self.edge_services.add(s)
                self.K_rem -= W
                if self.K_rem < 1:
                    break

    def reset(self):
        """Reset parameters updated by algorithms."""
        self.init_edge_services()
        self.cost_migration = 0
        self.cost_forwarding = 0
        self.cost = 0
        self.migrations = []
        # Count the times the offline_opt_recursion function is called.
        self.offline_opt_recursion_cnt = 0
        # Lookup table (LUT) of the offline_opt_recursion function.
        self.offline_opt_recursion_lut = defaultdict(lambda: (-1, []))
        # Count the times of LUT hits.
        self.offline_opt_recursion_lut_cnt = 0

    def run_belady(self, modified=False, alg_time_threshold=60):
        """Bélády's algorithm or the clairvoyant algorithm.

        The (unmodified) algorithm always migrate a missing service,
        and deletes the existing service whose next occurrence is
        farthest in the future. It is offline optimal (OPT) in the
        cache (or page placement) problem, where the forwarding cost
        is infinity, and the migration cost (M) is one.

        The modified version will forward the missing service, if its
        next occurrence is farther than those in the edge services.
        It should have the same cost performance as the offline optimal
        solution when the migration cost (M) is one. (But it might not
        the same as the offline optimal (OPT), since it might not have
        the most times of migrations.
        """
        self.reset()

        alg_name = 'Bélády'
        if modified:
            alg_name += ' Modified'
        # Estimated time (in seconds) needed to run the algorithm.
        # Seems that (K+1)-multiple search in list.index() is
        # optimized, so no multiplication by K in calculation below.
        alg_time = (self.N ** 2) / (2.5e8)
        if alg_time > 1:
            logging.info('alg_time={}'.format(alg_time))
        if alg_time > alg_time_threshold:
            # Mark invalid cost.
            logging.warning('{} is very likely to exceed the time '
                            'threshold so it is skipped. '
                            'Increase alg_time_threshold if you really '
                            'want to run it.'.format(alg_name))
            self.cost = None
            return

        # The sequence number of arrivals.
        n = 0
        for r in self.requests:
            n += 1
            W_new = self.W[r]
            logging.debug(
                'n={}, request={}, M={}, F={}, W={}, K_rem={}'
                .format(n, r, self.M[r], self.F[r], self.W[r], self.K_rem))

            migration = False
            forwarding = False
            if r in self.edge_services:
                # r is already hosted.
                logging.debug('result: hosted')
                continue
            elif W_new > self.K:
                forwarding = True
                self.cost_forwarding += self.F[r]
                logging.debug('result: forwarded (cannot fit in)'.format(r))
                continue
            else:
                # r is not hosted and can be hosted. Migrate it.
                migration = True
            assert migration
            assert not forwarding
            # Delete zero or more services until new service can fit in.
            ss_del = []
            if modified:
                # Note self.edge_services is modified and might need to be
                # restored in case the migration is aborted.
                self.edge_services.add(r)
            # Note the [s] concatenated to self.requests is to avoid ValueError
            # exception raised by list.index() when no such element is found.
            # Use a local K_rem in case the migration is aborted.
            K_rem = self.K_rem
            if K_rem < W_new:
                s_tuples = [(s, (self.requests + [s]).index(s, n))
                            for s in self.edge_services]
                sorted_s_tuples = sorted(
                    s_tuples,
                    key=lambda x: (x[1], x[0]))
            # Run and record the migration and deletion.
            while K_rem < W_new:
                s_tuple = sorted_s_tuples.pop()
                s_del = s_tuple[0]
                if s_del == r:
                    assert modified
                    # If the new service is to be deleted, then we abort the
                    # migration and switch to forwarding.
                    # self.K_rem remains unchanged, while
                    # self.edge_services needs to be restored.
                    forwarding = True
                    break
                ss_del.append(s_del)
                K_rem += self.W[s_del]
            if forwarding:
                assert modified
                self.edge_services.remove(r)
                self.cost_forwarding += self.F[r]
                logging.debug('result: forwarded (next request too late)'
                              .format(r))
                continue
            for s in ss_del:
                self.edge_services.remove(s)
            if not modified:
                self.edge_services.add(r)
            self.K_rem = K_rem - W_new
            assert self.K_rem == self.K - \
                sum(self.W[s] for s in self.edge_services)
            self.migrations.append((n, r, self.fmt_services(ss_del)))
            self.cost_migration += self.M[r]
            logging.debug('result: migrated. ({} deleted)'.format(ss_del))
        self.cost = self.cost_migration + self.cost_forwarding

    def run_RL(self, alg_time_threshold=60):
        """The RM-LRU (RL) Online algorithm.

        The algorithm combines a retrospective migration (RM) policy and a
        least recently used (LRU) policy for deletion.
        """
        self.reset()

        # Estimated time (in seconds) needed to run the algorithm.
        alg_time = self.N_unique * self.N / (1.6e7)
        if alg_time > 1:
            logging.info('alg_time={}'.format(alg_time))
        if alg_time > alg_time_threshold:
            # Mark invalid cost.
            logging.warning('RL is very likely to exceed the time '
                            'threshold so it is skipped. '
                            'Increase alg_time_threshold if you really '
                            'want to run it.')
            self.cost = None
            return

        # b_{i,j} = max_tau (sum x_j(l) - sum x_i(l))^+ for each service pair
        # Here sum is over the latest tau arrivals.
        # In the latest tau arrivals, i is hosted by the edge cloud,
        # and j is not hosted by the edge cloud.
        b = defaultdict(int)
        # The sequence numbers of the most recent ceil(2*M_i/F_i) arrivals.
        seqnums = dict()
        for s in self.services:
            qlen = math.ceil(2 * self.M[s] / self.F[s])
            seqnums[s] = deque([0]*qlen, maxlen=qlen)
        n = 0

        for r in self.requests:
            n += 1
            W_new = self.W[r]
            seqnums[r].append(n)
            logging.debug(
                'n={}, request={}, M={}, F={}, W={}, K_rem={}'
                .format(n, r, self.M[r], self.F[r], self.W[r], self.K_rem))
            # r == S_j (i^*)
            if r in self.edge_services:
                # r will be hosted by the edge cloud immediately. No cost.
                for b_key in b:
                    if ((b_key[0] == r) and
                            (b_key[1] not in self.edge_services)):
                        b[b_key] = max(0, b[b_key] - self.F[r])
                logging.debug('result: hosted')
                continue
            # Now we know r (i.e. S_j, i^*) is not hosted by the edge cloud.
            m = False  # Flag to indicate whether to migrate or not.
            for req in self.edge_services:
                # req (S_i) should be hosted by the edge cloud.
                b[(req, r)] += self.F[r]
                if b[(req, r)] >= self.M[req] + self.M[r]:
                    m = True
            if (not m) or (W_new > self.K):
                # r needs to be forwarded.
                logging.debug('result: forwarded')
                self.cost_forwarding += self.F[r]
                continue
            # Now we know r should and can be migrated.
            assert m and W_new <= self.K
            # Delete zero or more services until new service can fit in.
            # They are the ones in edge cloud with smallest sequence numbers
            # for the past 2M requests.
            ss_del = []
            if self.K_rem < W_new:
                # Build a list of tuples: (service, head of queue per service)
                s_tuples = [(s, seqnums[s][0]) for s in self.edge_services]
                sorted_s_tuples = sorted(
                    s_tuples,
                    key=lambda x: (x[1], x[0]),
                    reverse=True)
            # Run and record the migration and deletion.
            while self.K_rem < W_new:
                # Pop out the last one.
                s_tuple = sorted_s_tuples.pop()
                s_del = s_tuple[0]
                ss_del.append(s_del)
                self.edge_services.remove(s_del)
                self.K_rem += self.W[s_del]
            self.edge_services.add(r)
            self.K_rem -= W_new
            # Need to reset b_{i^*,j} = 0 for all j where i^* to be migrated,
            # and b_{i,j^*} = 0 for all i where j^* to be deleted.
            for b_key in b:
                if b_key[0] == r or b_key[1] in ss_del:
                    b[b_key] = 0
            self.migrations.append((n, r, self.fmt_services(ss_del)))
            self.cost_migration += self.M[r]
            logging.debug('result: migrated. ({} deleted)'.format(ss_del))
        self.cost = self.cost_migration + self.cost_forwarding

    def run_no_migration(self):
        self.reset()
        # If no migration happens...
        for r in self.requests:
            if r not in self.edge_services:
                self.cost_forwarding += self.F[r]
        assert self.cost_migration == 0
        self.cost = self.cost_forwarding

    def run_static(self):
        """Offline static algorithm.

        This algorithm migrates the most popular services in the beginning into
        the edge cloud, and forwards all other services.

        In the heterogeneous case, it uses dynamic programming to solve a
        back packing problem, which maximizes the total frequency counts of
        the most popular (i.e. wished) services, given that their total space
        is no more than the edge cloud capacity.
        """
        self.reset()
        # LUT for static_value().
        self.static_value_lut = defaultdict(lambda: (-1, []))
        self.static_value_lut_cnt = 0
        # Find the most popular services, and migrate them.
        for y in range(self.N_unique):
            for x in range(1, self.K + 1):
                self.static_value(x, y)

        v_ss_tuples = []
        for key in self.static_value_lut.keys():
            if 1 <= key[0] <= self.K and key[1] == self.N_unique - 1:
                v_ss_tuples.append(self.static_value_lut[key])
        # There might be multiple tuple reaching the maximum, but since their
        # value (which implies system's forwarding cost) and migration cost
        # are the same respectively, choosing either one should be fine.
        (v, ss) = max(
            v_ss_tuples,
            key=lambda x: (x[0], -self.mig_cost(x[1])))

        edge_services_wished = set(ss)
        self.migrations.append((
            1,
            self.fmt_services(edge_services_wished - self.edge_services),
            self.fmt_services(self.edge_services - edge_services_wished)))
        self.cost_migration = self.mig_cost(ss)
        # After migration, edge services are the ones we want.
        self.edge_services = edge_services_wished

        for r in self.requests:
            if r not in self.edge_services:
                self.cost_forwarding += self.F[r]

        self.cost = self.cost_migration + self.cost_forwarding

    def mig_cost(self, services):
        """Return the migration cost of services to be migrated.

        We compare services with the ones in self.edge_services, and
        find those really should be migrated, and then sum up their
        migration cost.

        :param services: an iterable of services to be migrated
        """
        edge_services_migrated = set(services) - self.edge_services
        return sum(self.M[es] for es in edge_services_migrated)

    def static_value(self, x, y):
        """Calculate the value function of DP for offline static alg.

        Value function V[x][y] := max total value with total weight x,
        while only using items in y.

        :param x: total weight constraint
        :param y: largest index (in self.sorted_requests) of services
                  which can be used
        :return (v, ss) tuple of value achieved and services selected
        """
        # Use the LUT.
        key = (x, y)
        if (self.static_value_lut[key])[0] >= 0:
            self.static_value_lut_cnt += 1
            return self.static_value_lut[key]

        # This may happen in recursive call. Mark the result invalid.
        if y < 0 or x < 0:
            res = (-math.inf, [])
            self.static_value_lut[key] = res
            return res

        (s, V) = self.sorted_requests_cnt[y]
        W = self.W[s]

        # y == 0 means only the first item is considered.
        # x == 0 means there is simply no space to fit in.
        if y == 0 or x == 0:
            if x >= W:
                res = (V, [s])
            else:
                res = (0, [])
            self.static_value_lut[key] = res
            return res
        assert y > 0 and x > 0
        (v, ss) = self.static_value(x - W, y - 1)
        res = max((v + V, ss + [s]), self.static_value(x, y - 1))
        self.static_value_lut[key] = res
        return res

    def print_migrations(self):
        format_str = '{0:>12}  {1!s:>25}  {2!s:>25}'
        logging.debug(format_str.format(
            'Sequence No.', 'Migrated Service ID(s)', 'Deleted Service ID(s)'))
        for migration in self.migrations:
            time = migration[0]
            migrated = migration[1]
            deleted = migration[2]
            logging.debug(format_str.format(time, migrated, deleted))

    def fmt_services(self, services):
        res = tuple(services)
        if len(res) == 1:
            return int(res[0])
        else:
            return res

    def get_cost(self):
        """Get the cost of the algorithm.

        This serves as the public API, and should be used instead of
        directly accessing the attribute.

        :return total cost (migration plus forwarding) of the algorithm
        """
        return self.cost


def parseNumRange(string):
    """Parse command line number range (e.g. 1-10, 2:5, 1, 2.33).

    :param string: command line argument string
    :return: a list of integers in the indicated range, or
             the original number if it is not a range.

    cf. http://stackoverflow.com/a/6512463/1166587
    """
    if string.find('-') >= 0:
        interval = string.split('-')
    elif string.find(':') >= 0:
        interval = string.split(':')
    else:
        interval = [string]
    start = interval[0]
    end = interval[-1]
    if start == end:
        # We know that string contains just a number.
        try:
            res = int(start)
        except ValueError:
            try:
                res = float(start)
                # If the floating point number equals an integer,
                # turn it into that integer for neat log filename.
                # (e.g. sim-M5.log instead sim-M5.0.log)
                if res == int(res):
                    res = int(res)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    '`{}\' is not a number or number range!'
                    .format(string))
        return [res]
    else:
        return list(range(int(start), int(end) + 1))


def main():
    """Main routine.
    """
    parser = argparse.ArgumentParser(
        description='Simulate edge cloud migration.')
    parser.add_argument('-N', dest='N', type=int, default=None,
                        help='number of requests from file '
                             'used in simulation')
    parser.add_argument('-K', dest='K',
                        type=parseNumRange, default=[5],
                        help='number of services hosted by edge cloud '
                             '(default: 5)')
    parser.add_argument('-M', dest='M',
                        type=float, default=None,
                        help='cost ratio of migration over forwarding '
                             '(leave out for heterogeneous system)')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='enable debug (default: disabled)')
    parser.add_argument('-f', dest='datafile', type=str, default=None,
                        help='data file containing the sequence of requests '
                        '(default: Google cluster data v1)')

    args = parser.parse_args()

    if args.datafile is None:
        datafile = 'traces/requests-job_id.dat'
        fname_str = 'v1'
    else:
        datafile = args.datafile
        fname_str = path.basename(datafile).rstrip('.dat')

    if args.M is None:
        if len(args.K) > 1:
            plot = True
            fname_str += 'N{}-het-K{}_{}'.format(
                args.N, args.K[0], args.K[-1])
        else:
            plot = False
            fname_str += 'N{}-het-K{}'.format(
                args.N, args.K[0])
    else:
        plot = False
        fname_str += 'N{}-K{}-M{}-hom'.format(
            args.N, args.K[0], args.M)

    # Configure logging.
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename='sim-' + fname_str + '.log',
            filemode='w')
    else:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format='%(message)s')

    labels = {
        'ST':  'Static',
        'BM':  'Bélády Mod',
        'RL':  'RL'}
    costs = OrderedDict([
        ('ST', []),
        ('BM', []),
        ('RL', [])])
    for k in args.K:
        if args.M is None:
            ec = EdgeCloud(datafile, K=k, N=args.N)
        else:
            ec = EdgeCloud(datafile, K=k, N=args.N, special=True, M=args.M)

        ec.run_static()
        costs['ST'].append(ec.get_cost())
        ec.print_migrations()
        logging.info('Total cost of {}: {}'
                     .format(labels['ST'], ec.get_cost()))

        ec.run_RL()
        costs['RL'].append(ec.get_cost())
        ec.print_migrations()
        logging.info('Total cost of RL: {}'.format(ec.get_cost()))

        ec.run_belady(modified=True)
        costs['BM'].append(ec.get_cost())
        ec.print_migrations()
        logging.info('Total cost of Bélády Modified: {}'
                     .format(ec.get_cost()))
    for key in costs.keys():
        logging.info('{:5}{}'.format(key, costs[key]))
    if not plot:
        return
    var = args.K
    var_str = 'K'
    styles = {
        'ST': 'k.-',
        'BM': 'bo-',
        'RL': 'r*-'}
    matplotlib.rcParams.update({'font.size': 16})
    for key in costs.keys():
        cost = np.array(costs[key], dtype=np.double)
        mask = np.isfinite(cost)
        var = np.array(var, dtype=np.uint32)
        plt.plot(var[mask], cost[mask],
                 styles[key], label=labels[key],
                 linewidth=2.0)
    plt.xlabel(var_str)
    plt.ylabel('Cost')
    plt.title('Heterogeneous System')
    plt.legend(loc='best')
    fname = 'fig-' + fname_str + '.pdf'
    plt.savefig(fname, bbox_inches='tight')

if __name__ == '__main__':
    main()
