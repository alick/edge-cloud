#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulation of Edge Cloud Migration

# defaultdict provides default values for missing keys
from collections import defaultdict, deque
from functools import partial
import argparse
import logging
import sys
import math


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
            raise Exception('The parameter K should be a postive integer.')
        if M >= 1:
            self.M = float(M)
        else:
            raise Exception('The parameter M should be at least 1.')
        if N is None or N >= 1:
            self.N = int(N)
        else:
            raise Exception('The parameter N should be '
                            'a postive integer or None.')
        self.requests = []
        with open(path_to_file, 'r') as f:
            if self.N is None:
                for line in f:
                    r = int(line)  # one request specified by its service id
                    self.requests.append(r)
            else:
                for x in range(self.N):
                    line = next(f)
                    r = int(line)
                    self.requests.append(r)
        logging.debug('# services: {0}'.format(len(self.requests)))

        requests_cnt = defaultdict(int)  # a dict with default integer value 0
        for r in self.requests:
            requests_cnt[r] += 1
        logging.debug('# unique services: {0}'.format(len(requests_cnt)))

        # The requests sorted by values and then keys.
        self.sorted_requests_cnt = sorted(
            requests_cnt.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True)

        # The requests sorted by keys (IDs) in ascending order.
        self.sorted_requests = sorted(requests_cnt.keys())

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

    def run_belady(self):
        self.reset()
        raise Exception('Unimplemented!')

    def run_RL(self):
        """The RM-LRU (RL) Online algorithm.

        The algorithm combines a retrospective migration (RM) policy and a
        least recently used (LRU) policy for deletion.
        """
        self.reset()

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
            # It is the one in edge cloud with smallest sequece number for the
            # past 2M requests
            svc_del = None
            for svc in self.edge_services:
                if len(seqnums[svc]) < 2*self.M:
                    # We have not seen >= 2M requests from svc from beginning.
                    # Set it to 0 so that we prefer it.
                    seqnum_2M = 0
                else:
                    seqnum_2M = seqnums[svc][0]
                # Initialize svc_del with the first service we encounter.
                if svc_del is None:
                    svc_del = svc
                    svc_del_seq = seqnum_2M
                    continue
                if seqnum_2M < svc_del_seq:
                    svc_del = svc
                    svc_del_seq = seqnum_2M
            assert svc_del in self.edge_services
            assert svc_del_seq < n
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

    def get_cost(self):
        return self.cost

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
        assert self.cost_forwarding == len(self.requests) - sum(
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate edge cloud migration.')
    parser.add_argument('-N', dest='N', type=int, default=None,
                        help='number of requests from file used in simulation')
    parser.add_argument('-K', dest='K', type=int, default=5,
                        help='number of services hosted by edge cloud '
                             '(default: 5)')
    parser.add_argument('-M', dest='M', type=float, default=5,
                        help='cost ratio of migration over forwarding '
                             '(default: 5)')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='enable debug (default: disabled)')

    args = parser.parse_args()

    # Configure logging.
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename='sim-N{}-K{}-M{}.log'.format(
                args.N, args.K, args.M),
            filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
            stream=sys.stdout, format='%(message)s')

    ec = EdgeCloud('traces/requests-job_id.dat', K=args.K, M=args.M,
                   N=args.N)

    ec.run_RL()
    ec.print_migrations()
    logging.info('Total cost for RL online algorithm: {}.'.format(ec.get_cost()))

    ec.run_no_migration()
    logging.info('Total cost with no migration: {}.'.format(ec.get_cost()))

    ec.run_static()
    ec.print_migrations()
    logging.info('Total cost for offline static algorithm: {}.'.format(ec.get_cost()))

    ec.run_belady()
