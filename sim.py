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


class EdgeCloud():
    """The EdgeCloud class holds all the magic."""

    def __init__(self, path_to_file, K=5, M=5, N=None):
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
        if M >= 1:
            self.M = float(M)
        else:
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
        assert len(self.services) == self.N_unique

        if self.N_unique <= self.K:
            logging.warning('WARN: Storage can hold all possible '
                            'services!')
            self.K = self.N_unique

        logging.info('No. requests: N={0}'.format(self.N))
        logging.info('No. unique services: |S|={0}'
                     .format(self.N_unique))
        logging.info('No. edge services: K={0}'.format(self.K))
        logging.info('Cost ratio: M={0}'.format(self.M))

        # Python 3.5 offers math.inf, which is equivalent to float('inf').
        try:
            math.inf
        except AttributeError:
            math.inf = float('inf')

        self.reset()

    def reset(self):
        """Reset parameters updated by algorithms."""
        # Fill in the initial K edge cloud services.
        # We use the K services with lowest id.
        self.edge_services = set(self.sorted_requests[0:self.K])
        self.cost_migration = 0
        self.cost_forwarding = 0
        self.cost = 0
        self.migrations = []
        self.requests_seen = []
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
            logging.debug('n={}, request={}'.format(n, r))

            migration = False
            if r not in self.edge_services:
                # r is not hosted. Migrate it.
                migration = True
            else:
                # r is already hosted.
                logging.debug('result: hosted')
                continue
            assert migration
            # Find the service to be deleted.
            # Note the [s] concatenated to self.requests is to avoid ValueError
            # exception raised by list.index() when no such element is found.
            if modified:
                self.edge_services.add(r)
            svc_tuples = [(s, (self.requests + [s]).index(s, n))
                          for s in self.edge_services]
            svc_del = max(svc_tuples, key=lambda x: (x[1], x[0]))[0]
            assert svc_del in self.edge_services
            # Run and record the migration and deletion.
            self.edge_services.remove(svc_del)
            if r != svc_del:
                self.edge_services.add(r)
                self.migrations.append((n, r, svc_del))
                self.cost_migration += self.M
                logging.debug('result: migrated. ({} deleted)'.format(svc_del))
            else:
                assert modified
                self.cost_forwarding += 1
                logging.debug('result: forwarded'.format(r))
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
        # The sequence numbers of the most recent 2*M arrivals for all services
        seqnums = defaultdict(
            lambda: deque([0]*2*math.ceil(self.M), maxlen=2*math.ceil(self.M)))
        # The sequence number of the latest migration for each service.
        seqnum_mig = defaultdict(int)
        # The sequence number of the latest deletion for each service.
        seqnum_del = defaultdict(int)
        n = 0

        for r in self.requests:
            n += 1
            self.requests_seen.append(r)
            assert len(self.requests_seen) == n
            seqnums[r].append(n)
            logging.debug('n={}, request={}'.format(n, r))
            # r == S_j
            if r in self.edge_services:
                # r will be hosted by the edge cloud immediately. No cost.
                for b_key in b:
                    if ((b_key[0] == r) and
                            (b_key[1] not in self.edge_services)):
                        b[b_key] = max(0, b[b_key] - 1)
                logging.debug('result: hosted')
                continue
            # Now we know r (i.e. S_j, i^*) is not hosted by the edge cloud.
            migration = False
            for req in self.edge_services:
                # req (S_i) should be hosted by the edge cloud.
                b[(req, r)] += 1
                if b[(req, r)] >= 2*self.M:
                    migration = True
            if not migration:
                # r needs to be forwarded.
                logging.debug('result: forwarded')
                self.cost_forwarding += 1
                continue
            assert migration is True
            # Find the service to be deleted.
            # It is the one in edge cloud with smallest sequence number for the
            # past 2M requests
            # Build a list of tuples: (service, head of queue per service)
            svc_tuples = [(s, seqnums[s][0]) for s in self.edge_services]
            svc_del = min(svc_tuples, key=lambda x: (x[1], x[0]))[0]
            assert svc_del in self.edge_services
            # Run and record the migration and deletion.
            self.edge_services.remove(svc_del)
            self.edge_services.add(r)
            assert r in self.edge_services
            assert svc_del not in self.edge_services
            # Need to reset b_{i^*,j} = 0 for all j where i^* to be migrated,
            # and b_{i,j^*} = 0 for all i where j^* to be deleted.
            for b_key in b:
                if b_key[0] == r or b_key[1] == svc_del:
                    b[b_key] = 0
            self.migrations.append((n, r, svc_del))
            self.cost_migration += self.M
            logging.debug('result: migrated. ({} deleted)'.format(svc_del))
            seqnum_mig[r] = n
            seqnum_del[svc_del] = n
        self.cost = self.cost_migration + self.cost_forwarding

    def run_no_migration(self):
        self.reset()
        # If no migration happens...
        for r in self.requests:
            if r not in self.edge_services:
                self.cost_forwarding += 1
        assert self.cost_migration == 0
        self.cost = self.cost_forwarding

    def run_static(self):
        """Offline static algorithm.

        This algorithm migrates the K most popular services once and for all.
        """
        self.reset()
        # Find the K most popular services, and migrate them.
        edge_services_wished = set(
            x[0] for x in self.sorted_requests_cnt[0:self.K])
        edge_services_migrated = edge_services_wished - self.edge_services
        self.migrations.append((
            1,
            tuple(edge_services_migrated),
            tuple(self.edge_services - edge_services_wished)))
        self.cost_migration = self.M * len(edge_services_migrated)
        # After migration, edge services are the ones we want.
        self.cost_forwarding = 0
        self.edge_services = edge_services_wished
        for r in self.requests:
            if r not in self.edge_services:
                self.cost_forwarding += 1
        assert self.cost_forwarding == self.N - sum(
            x[1] for x in self.sorted_requests_cnt[0:self.K])
        self.cost = self.cost_migration + self.cost_forwarding

    def print_migrations(self):
        format_str = '{0:>12}  {1!s:>25}  {2!s:>25}'
        logging.debug(format_str.format(
            'Sequence No.', 'Migrated Service ID(s)', 'Deleted Service ID(s)'))
        for migration in self.migrations:
            time = migration[0]
            migrated = migration[1]
            deleted = migration[2]
            logging.debug(format_str.format(time, migrated, deleted))

    def fmt_svc_set(self, svc_set):
        res = tuple(svc_set)
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
                        type=parseNumRange, default=[5],
                        help='cost ratio of migration over forwarding '
                             '(default: 5)')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='enable debug (default: disabled)')

    args = parser.parse_args()

    if len(args.M) == 1 and len(args.K) == 1:
        plot = False
    elif len(args.M) == 1 or len(args.K) == 1:
        # One of K and M is a list of integers.
        plot = True
    else:
        # Both K and M are lists. Not supported.
        raise Error('K and M are both ranges. Not supported!')

    # Configure logging.
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename='sim-N{}-K{}-M{}.log'.format(
                args.N, args.K, args.M),
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
        ec = EdgeCloud('traces/requests-job_id.dat',
                       K=k, N=args.N)

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
    plt.legend(loc='best')
    fname = 'fig-N{}-het-{}{}_{}.pdf'.format(
        args.N, var_str, var[0], var[-1])
    plt.savefig(fname, bbox_inches='tight')

if __name__ == '__main__':
    main()
