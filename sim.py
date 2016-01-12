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
import random
import re
from math import sqrt
SPINE_COLOR = 'gray'


class EdgeCloud():
    """The EdgeCloud class holds all the magic."""

    def __init__(self, requests, K=5, M=5,
                 max_time=60, max_mem=1e9):
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
        self.max_time = max_time
        self.max_mem = max_mem
        self.requests = requests
        self.N = len(self.requests)

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

    def run(self, alg):
        if alg == 'BM':
            self.run_belady(modified=True)
        elif alg == 'ST':
            self.run_static()
        elif alg == 'IT':
            self.run_offline_iterative()
        elif alg == 'RN':
            self.run_online_randomized()
        elif alg == 'RL':
            self.run_RL()
        elif alg == 'OPT':
            self.run_offline_opt()
        elif alg == 'NM':
            self.run_no_migration()
        elif alg == 'BE':
            self.run_belady(modified=False)
        else:
            raise Exception('Unknown algorithm: {}'.format(alg))

    def run_belady(self, modified=False, max_time=None):
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
        if max_time is None:
            max_time = self.max_time
        # Estimated time (in seconds) needed to run the algorithm.
        # Seems that (K+1)-multiple search in list.index() is
        # optimized, so no multiplication by K in calculation below.
        alg_time = (self.N ** 2) / (2.5e8)
        if alg_time > 1:
            logging.info('alg_time={}'.format(alg_time))
        if alg_time > max_time:
            # Mark invalid cost.
            logging.warning('{} is very likely to exceed the time '
                            'threshold so it is skipped. '
                            'Increase max_time if you really '
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

    def run_online_randomized(self):
        """The Online Randomized algorithm.

        The algorithm migrates a service with probability 1/M, and
        deletes a random service in the edge cloud uniformly.
        """
        self.reset()

        n = 0
        for r in self.requests:
            n += 1
            logging.debug('n={}, request={}'.format(n, r))
            if r in self.edge_services:
                # r will be hosted by the edge cloud immediately. No cost.
                logging.debug('result: hosted')
                continue
            if random.random() > 1 / self.M:
                # No migration. Forwarding.
                self.cost_forwarding += 1
                logging.debug('result: forwarded')
                continue
            # Find a service to delete randomly.
            s_del = random.choice(tuple(self.edge_services))
            self.edge_services.remove(s_del)
            self.edge_services.add(r)
            self.migrations.append((n, r, s_del))
            self.cost_migration += self.M
            logging.debug('result: migrated. ({} deleted)'.format(s_del))
        self.cost = self.cost_migration + self.cost_forwarding

    def run_RL(self, max_time=None):
        """The RM-LRU (RL) Online algorithm.

        The algorithm combines a retrospective migration (RM) policy and a
        least recently used (LRU) policy for deletion.
        """
        self.reset()

        if max_time is None:
            max_time = self.max_time
        # Estimated time (in seconds) needed to run the algorithm.
        alg_time = self.N_unique * self.N / (1.6e7)
        if alg_time > 1:
            logging.info('alg_time={}'.format(alg_time))
        if alg_time > max_time:
            # Mark invalid cost.
            logging.warning('RL is very likely to exceed the time '
                            'threshold so it is skipped. '
                            'Increase max_time if you really '
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

    def run_offline_opt(self, max_time=None, max_mem=None):
        """Offline optimal (OPT) algorithm.

        The algorithm calls the offline_opt_recursion routine to find the
        optimal solution. Its time complexity is roughly O(K^N).

        :param max_time: estimated allowable time to run the
                                   OPT algorithm (default 60s)
        """

        self.reset()

        # Whether to record n in the debug log to indicate the progress.
        log_n = False
        if max_time is None:
            max_time = self.max_time
        # Estimated time (in seconds) needed to run the algorithm.
        alg_time = (self.N_unique ** self.K) * self.K * self.N / (1.5e5)
        if alg_time > 1:
            logging.info('alg_time={}'.format(alg_time))
        if alg_time > max_time:
            # Mark invalid cost.
            logging.warning('OPT is very likely to exceed the time '
                            'threshold so it is skipped. '
                            'Increase max_time if you really '
                            'want to run it.')
            self.cost = None
            return
        elif alg_time > (max_time * 0.5):
            log_n = True
            logging.warning('OPT can be quite time consuming! '
                            'Progress will be displayed.')

        if max_mem is None:
            max_mem = self.max_mem
        # LUT size assuming each (key, value) pair takes up 100 Bytes.
        alg_mem = (self.N_unique ** self.K) * 2 * 100
        if alg_mem > 0.1 * max_mem:
            logging.info('alg_mem={:.2e}'.format(alg_mem))
        if alg_mem > max_mem:
            logging.warning('OPT is very likely to exceed the memory '
                            'threshold so it is skipped. '
                            'Increase max_mem if you really '
                            'want to run it.')
            self.cost = None
            return

        # Pre-calculate offline_opt_recursion(n, es) for all n in 1:N-1 and all
        # possible set of edge services, so that LUT caches the intermediate
        # results.
        for n in range(self.N + 1):
            if log_n:
                logging.info('n={}'.format(n))
            for es in itertools.combinations(self.services, self.K):
                self.offline_opt_recursion(n=n, edge_services=set(es))
            # LUT prev <= current
            for key in self.offline_opt_recursion_lut.keys():
                if key[0] == 1:
                    self.offline_opt_recursion_lut[(0,) + key[1:]] = \
                        self.offline_opt_recursion_lut[key]
                    self.offline_opt_recursion_lut[key] = (-1, [])
        cost_mig_svc_tuples = []
        for key in self.offline_opt_recursion_lut.keys():
            if key[0] == 0:
                cost_mig_svc_tuples.append(
                    self.offline_opt_recursion_lut[key] + key[1:])
        # Find the one with minimum cost.
        # If tie: select the one with maximum number of migrations,
        # If still tie: select the one with smallest ID(s).
        cost_mig_svc_tuple = min(
            cost_mig_svc_tuples,
            key=lambda x: (x[0], -len(x[1])) + x[2:])
        (self.cost, self.migrations) = cost_mig_svc_tuple[0:2]

        logging.debug('Function offline_opt_recursion() was called {} times.'
                      .format(self.offline_opt_recursion_cnt))
        logging.debug('LUT was hit {} times.'
                      .format(self.offline_opt_recursion_lut_cnt))

    def offline_opt_recursion(self, n, edge_services):
        """Offline optimal (OPT) recursive routine.

        :param n: sequence number of request arrival
        :param edge_services: the edge_services after the n-th arrival
        :return (cost, migrations)
        """
        self.offline_opt_recursion_cnt += 1
        # First, check whether it has been calculated before.
        # The key of LUT is a tuple of the sequence number followed by all the
        # edge services sorted by their IDs in the ascending order.
        # The value is a tuple of the cost calculated, followed by the list of
        # corresponding migrations.
        key = (1,) + tuple(sorted(edge_services))
        if (self.offline_opt_recursion_lut[key])[0] >= 0:
            self.offline_opt_recursion_lut_cnt += 1
            return self.offline_opt_recursion_lut[key]

        # Initial cost is zero with predefined edge_services, otherwise
        # mark it infinity.
        if (n <= 0):
            if edge_services == self.edge_services:
                res = (0, [])
            else:
                res = (math.inf, [])
            self.offline_opt_recursion_lut[key] = res
            key_tmp = (0,) + key[1:]
            self.offline_opt_recursion_lut[key_tmp] = res
            return res

        # Otherwise, deal with the n-th (n >= 1) arrival.
        # Note that Python lists count from 0.
        r = self.requests[n - 1]
        if r not in edge_services:
            # forwarding, no migration
            key_tmp = (0,) + tuple(sorted(edge_services))
            (c, m) = self.offline_opt_recursion_lut[key_tmp]
            res = (c + 1, m)
            self.offline_opt_recursion_lut[key] = res
            return res
        assert r in edge_services
        svc_tuples = []  # a list of (service to be deleted, cost)
        # Find cost of migrating r & deleting one service in previous
        # configuration.
        for s in self.services - edge_services:
            es = edge_services.copy()
            es.remove(r)
            es.add(s)
            key_tmp = (0,) + tuple(sorted(es))
            (c, m) = self.offline_opt_recursion_lut[key_tmp]
            cost = self.M + c
            migrations = m + [(n, r, s)]
            svc_tuples.append((s, cost, migrations))
        # Find cost of hosting r
        key_tmp = (0,) + tuple(sorted(edge_services))
        (c, m) = self.offline_opt_recursion_lut[key_tmp]
        svc_tuples.append((0, c, m))  # 0 is a fake service ID
        # Find the one with minimum cost.
        # If tie: select the one with maximum number of migrations,
        # If still tie: select the one with lowest ID.
        svc_tuple = min(svc_tuples, key=lambda x: (x[1], -len(x[2]), x[0]))
        res = (svc_tuple[1], svc_tuple[2])
        self.offline_opt_recursion_lut[key] = res
        return res

    def run_offline_iterative(self, max_time=None, max_mem=None):
        """Offline iterative algorithm.

        The algorithm runs offline optimal algorithm iteratively, each time
        considering one more space for services in the edge cloud.
        """
        self.reset()

        if max_time is None:
            max_time = self.max_time
        # Estimated time (in seconds) needed to run the algorithm.
        alg_time = (self.N_unique ** 2) * self.K * self.N / (1e7)
        if alg_time > 1:
            logging.info('alg_time={:.3f}'.format(alg_time))
        if alg_time > max_time:
            # Mark invalid cost.
            logging.warning('Offline Iterative is very likely to '
                            'exceed the time '
                            'threshold so it is skipped. '
                            'Increase max_time if you really '
                            'want to run it.')
            self.cost = None
            return

        if max_mem is None:
            max_mem = self.max_mem
        # LUT size assuming each (key, value) pair takes up 100 Bytes,
        # plus the size of edge_services_matrix
        alg_mem = (self.N_unique) * 2 * 100 + self.N * self.K * 4
        if alg_mem > 0.1 * max_mem:
            logging.info('alg_mem={:.2e}'.format(alg_mem))
        if alg_mem > max_mem:
            logging.warning('OPT is very likely to exceed the memory '
                            'threshold so it is skipped. '
                            'Increase max_mem if you really '
                            'want to run it.')
            self.cost = None
            return

        # LUT for offline_iterative_cost function.
        self.offline_iterative_cost_lut = defaultdict(lambda: (-1, []))
        # Matrix whose (k, n)-th element denotes the edge service in k-th
        # position after n-th arrival.
        self.edge_services_matrix = np.zeros((self.K, self.N + 1),
                                             dtype=np.uint32)
        self.edge_services_matrix[:, 0] = sorted(self.edge_services)

        for k in range(1, self.K + 1):
            for n in range(self.N + 1):
                for s in self.services:
                    self.offline_iterative_cost(k, n, s)
                # LUT prev <= current
                for key in self.offline_iterative_cost_lut:
                    if key[0] == 1:
                        self.offline_iterative_cost_lut[(0,) + key[1:]] = \
                            self.offline_iterative_cost_lut[key]
                        self.offline_iterative_cost_lut[key] = (-1, [])
            c_e_m_tuples = []
            for key in self.offline_iterative_cost_lut.keys():
                if key[0] == 0:
                    c_e_m_tuples.append(
                        self.offline_iterative_cost_lut[key])
            # Find the one with minimum cost.
            # If tie: select the one with maximum number of migrations,
            # If still tie: select the one whose last service ID is smallest.
            (c, e, m) = min(c_e_m_tuples,
                            key=lambda x: (x[0], -x[2], x[1][-1]))
            self.edge_services_matrix[k - 1, :] = e
            if k == 1:
                logging.debug('k=1, cost={}, e={}, mig_cnt={}'.format(c, e, m))
        if self.N <= 4:
            logging.debug('matrix=\n{}'.format(self.edge_services_matrix))
        # Note that a service might appear in more than one unit of storage.
        # The eventual migration events need to be calculated after all
        # storage are considered.
        edge_services_prev = set(self.edge_services_matrix[:, 0])
        assert edge_services_prev == self.edge_services
        for n in range(1, self.N + 1):
            edge_services_cur = set(self.edge_services_matrix[:, n])
            edge_services_migrated = edge_services_cur - edge_services_prev
            edge_services_deleted = edge_services_prev - edge_services_cur
            if len(edge_services_migrated) or len(edge_services_deleted):
                self.migrations.append((
                    n,
                    self.fmt_svc_set(edge_services_migrated),
                    self.fmt_svc_set(edge_services_deleted)))
            # Migrations should be at most once at a time.
            if len(edge_services_migrated) > 1:
                logging.warning('I cannot believe it!')
                logging.warning('n={}'.format(n))
            self.cost_migration += len(edge_services_migrated) * self.M
            if self.requests[n - 1] not in edge_services_cur:
                self.cost_forwarding += 1
            edge_services_prev = edge_services_cur
        self.cost = self.cost_migration + self.cost_forwarding

    def offline_iterative_cost(self, k, n, s):
        """Calculate the cost of offline iterative algorithm with LUT.

        :param k: position of the service s in the edge cloud storage (1..K)
        :param n: sequence number of the request arrival
        :param s: newly added edge service
        :return (cost, edge_service_chain, mig_cnt)
                  minimum cost after the n-th arrival,
                  the corresponding edge service chain from the start
                  (0-th arrival) to the n-th arrival (inclusive), and
                  its migration counts, given that
                  L_k stores the service s after the n-th arrival.
        """

        key = (1, s)
        key_prev = (0, s)

        if (n <= 0):
            if s == self.edge_services_matrix[k - 1, 0]:
                res = (0, [s], 0)
            else:
                res = (math.inf, [s], 0)
            self.offline_iterative_cost_lut[key] = res
            # Initialize LUT prev too to avoid dict size change at runtime.
            self.offline_iterative_cost_lut[key_prev] = res
            return res

        r = self.requests[n - 1]
        (c, e, m) = self.offline_iterative_cost_lut[key_prev]
        if s != r:
            # r is forwarded if r is not in storage 1 to k-1,
            # otherwise r is hosted for sure.
            # In both cases, s should be here after the previous arrival,
            # and stay here after the n-th arrival.
            if r not in self.edge_services_matrix[0:k - 1, n]:
                res = (c + 1, e + [s], m)
            else:
                res = (c, e + [s], m)
            self.offline_iterative_cost_lut[key] = res
            return res
        assert r == s
        # Now we know s is the same as r, which means s was here after (n-1)-th
        # arrival already, or there is a migration after n-th arrival.
        svc_tuples = [(c, e + [s], m)]
        for svc in self.services - set([s]):
            key_tmp = (0, svc)
            (c, e, m) = self.offline_iterative_cost_lut[key_tmp]
            svc_tuples.append((c + self.M, e + [s], m + 1))
        # Find the one with minimum cost.
        # If tie: select the one with maximum number of migrations,
        # If still tie: select the deleted service with smallest ID.
        res = min(svc_tuples, key=lambda x: (x[0], -x[2], x[1][-2]))
        self.offline_iterative_cost_lut[key] = res
        return res

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


def get_requests(datafile, N=None):
    """Get a list of requests from the data file.

    :param datafile: data file containing one (integer) request per line
    :param N: only the first N requests needed
    :return: a list of requests
    """
    requests = []
    with open(datafile) as f:
        if N is None:
            for line in f:
                r = int(line)
                requests.append(r)
        else:
            for x in range(N):
                line = next(f)
                r = int(line)
                requests.append(r)
    return requests


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


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Code adapted from https://nipunbatra.github.io/2014/08/latexify/ ,
    which is adapted from
    http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1) / 2     # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'axes.labelsize': 8,   # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8,        # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              # 'text.usetex': True,
              # 'text.latex.preamble': ['\usepackage{gensymb}'],
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


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
    parser.add_argument('--max-time', dest='max_time', type=int, default=60,
                        help='maximum time in seconds to run each algorithm '
                             '(default: 60)')
    parser.add_argument('--max-mem', dest='max_mem', type=float, default=1e9,
                        help='maximum memory in bytes to run each algorithm '
                             '(default: 1e9)')
    parser.add_argument('-f', dest='datafile', default=None, nargs='*',
                        help='data file(s) with the sequence of requests '
                        '(default: Google cluster data v1)')
    parser.add_argument('-l', '--load', dest='load', action='store_true',
                        help='load previously saved result directly '
                        '(default: False)')

    args = parser.parse_args()

    if args.datafile is None or len(args.datafile) <= 0:
        args.datafile = ['traces/requests-job_id.dat']
        fname_str = 'v1-avg'
    else:
        # Get the part without directory nor suffix for each input file.
        basename = [path.splitext(path.basename(f))[0] for f in args.datafile]
        if len(basename) == 1:
            fname_str = basename[0]
        else:
            matches = [re.match('\D*(\d+)(.*)', x) for x in basename]
            if None in matches:
                raise Exception('Filenames should contain numbers. '
                                '(e.g. 0.dat, part-00-of-10.dat)')
            start = matches[0].group(1)
            end = matches[-1].group(1)
            other = matches[0].group(2)
            fname_str = '{}-{}{}'.format(start, end, other)

    if args.N is not None:
        fname_str += '-N{}'.format(args.N)

    if len(args.M) == 1 and len(args.K) == 1:
        plot = False
        fname_str += '-K{}-M{}'.format(args.K[0], args.M[0])
    elif len(args.M) == 1 or len(args.K) == 1:
        # One of K and M is a list of integers.
        if len(args.M) > len(args.K):
            var = args.M
            con = args.K[0]
            var_str = 'M'
            con_str = 'K'
        else:
            var = args.K
            con = args.M[0]
            var_str = 'K'
            con_str = 'M'
        plot = True
        fname_str += '-{}{}-{}{}_{}'.format(
            con_str, con, var_str, var[0], var[-1])
    else:
        # Both K and M are lists. Not supported.
        raise Error('K and M are both ranges. Not supported!')
    N_var = max(len(args.M), len(args.K))

    # Configure logging.
    if args.debug:
        assert not plot
        assert args.N is not None
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename='sim-N{}-K{}-M{}.log'.format(
                args.N, args.K[0], args.M[0]),
            filemode='w')
    else:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format='%(message)s')

    labels = OrderedDict([
        ('RN', 'Randomized'),
        ('BM', 'Belady Mod'),
        ('ST', 'Static'),
        ('IT', 'Iterative'),
        ('RL', 'RL'),
        ])

    N_file = len(args.datafile)
    npzfile = 'dat-' + fname_str + '.npz'
    if not args.load:
        requests_chunks = []  # a list of lists
        if N_file == 1:
            N_chunk = 10
            chunk_step = 100000
            assert args.N is not None
            requests = get_requests(args.datafile[0])
            for n_chunk in range(N_chunk):
                base = n_chunk * chunk_step
                requests_chunks.append([])
                requests_chunks[n_chunk] = requests[base:base+args.N]
        else:
            for datafile in args.datafile:
                requests_chunks.append(get_requests(datafile))
            N_chunk = len(requests_chunks)
            assert N_chunk == N_file
        A = np.ones((N_chunk, N_var), dtype=np.double) * np.nan
        costs = OrderedDict([
            ('RN', A.copy()),
            ('BM', A.copy()),
            ('ST', A.copy()),
            ('IT', A.copy()),
            ('RL', A.copy()),
            ])
        del A
        for n_chunk in range(N_chunk):
            n_var = 0
            for k in args.K:
                for m in args.M:
                    ec = EdgeCloud(requests_chunks[n_chunk],
                                   K=k, M=m,
                                   max_time=args.max_time,
                                   max_mem=args.max_mem)
                    for alg in labels.keys():
                        if alg != 'RN':
                            N_run = 1
                        else:
                            N_run = 10
                        cost_array = np.array([np.nan] * N_run)
                        for n in range(N_run):
                            ec.run(alg)
                            ec.print_migrations()
                            logging.debug('Total cost of {}: {}'
                                          .format(labels[alg], ec.get_cost()))
                            cost_array[n] = ec.get_cost()
                        costs[alg][n_chunk, n_var] = np.nanmean(cost_array)
                    n_var += 1
        np.savez(npzfile, **costs)
    else:
        costs = np.load(npzfile)
    for key in labels.keys():
        logging.info('{}\n{}'.format(key, costs[key]))
    if not plot:
        return
    styles = {
        'RN': 'mx-',
        'BM': 'bo-',
        'ST': 'kd-',
        'IT': 'g^-',
        'RL': 'r*-',
        }
    var = np.array(var, dtype=np.uint32)
    latexify()
    for key in labels.keys():
        costs_mat = costs[key]
        cost = np.ravel(np.nanmean(costs_mat, axis=0))
        mask = np.isfinite(cost)
        plt.plot(var[mask], cost[mask],
                 styles[key], label=labels[key])
    plt.xlabel(var_str)
    plt.ylabel('Cost')
    # Dirty hack to not let legend cover data points.
    if args.N == 1000:
        plt.ylim(100, 950)
    elif args.N == 10000:
        plt.ylim(0, 13000)
    plt.title(con_str + '={}'.format(con))
    plt.legend(loc='best')
    format_axes(plt.gca())
    fname = 'fig-' + fname_str + '.pdf'
    plt.savefig(fname, bbox_inches='tight')

if __name__ == '__main__':
    main()
