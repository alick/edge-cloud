#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulation of Edge Cloud Migration

# defaultdict provides default values for missing keys
from collections import defaultdict, deque
from functools import partial


class EdgeCloud():
    """The EdgeCloud class holds all the magic."""

    def __init__(self, path_to_file, K=5, M=5):
        """Initialize with a file containing the sequence of requests.

        :param path_to_file: string to specify the path to the file
        :param K: number of servers hosted by edge cloud (default: 5)
        :param M: cost ratio of migration over forwarding (default: 5)
        """
        if K >= 1:
            self.K = K
        else:
            raise Exception('The parameter K should be a postive integer.')
        if M >= 1:
            self.M = M
        else:
            raise Exception('The parameter M should be at least 1.')
        self.requests = []
        with open(path_to_file, 'r') as f:
            for line in f:
                r = int(line)  # one request specified by its service id
                self.requests.append(r)
        print('# services: {0}'.format(len(self.requests)))

        requests_cnt = defaultdict(int)  # a dict with default integer value 0
        for r in self.requests:
            requests_cnt[r] += 1
        print('# unique services: {0}'.format(len(requests_cnt)))

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

        # b_{i,j} = max_tau (sum x_j(l) - sum x_i(l))^+ for each pair of service
        # Here sum is over the latest tau arrivals.
        # In the latest tau arrivals, i is hosted by the edge cloud,
        # and j is not hosted by the edge cloud.
        b = defaultdict(int)
        # The sequence numbers of the most recent 2*M arrivals for all services
        seqnums = defaultdict(partial(deque, maxlen=2*self.M))
        # The sequence number of the latest migration for each service.
        seqnum_mig = defaultdict(int)
        # The sequence number of the latest deletion for each service.
        seqnum_del = defaultdict(int)
        seqnum_cur = 0

        for r in self.requests:
            seqnum_cur += 1
            self.requests_seen.append(r)
            assert len(self.requests_seen) == seqnum_cur
            seqnums[r].append(seqnum_cur)
            print('n={}, request={}'.format(seqnum_cur, r))
            # r == S_j
            if r in self.edge_services:
                # r will be hosted by the edge cloud immediately. No cost.
                for b_key in b:
                    if ((b_key[0] == r) and
                        (b_key[1] not in self.edge_services)):
                        b[b_key] = max(0, b[b_key] - 1)
                print('result: hosted')
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
                print('result: forwarded')
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
            assert svc_del_seq < seqnum_cur
            # Run and record the migration and deletion.
            self.edge_services.remove(svc_del)
            self.edge_services.add(r)
            assert r in self.edge_services
            assert svc_del not in self.edge_services
            # Need to reset b_{i^*,j} = 0 for all j
            for b_key in b:
                if b_key[0] == r:
                    b[b_key] = 0
            self.migrations.append((seqnum_cur, r, svc_del))
            self.cost_migration += self.M
            print('result: migrated. ({} deleted)'.format(svc_del))
            seqnum_mig[r] = seqnum_cur
            seqnum_del[svc_del] = seqnum_cur
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
        print('Time slot\tMigrated Service ID(s)\tDeleted Service ID(s)')
        for migration in self.migrations:
            time = migration[0]
            migrated = migration[1]
            deleted = migration[2]
            print('{}\t\t{}\t{}'.format(time, migrated, deleted))


if __name__ == '__main__':
    ec = EdgeCloud('traces/requests-job_id.dat', K=5, M=5)

    ec.run_RL()
    ec.print_migrations()
    print('Total cost for RL online algorithm: {}.'.format(ec.get_cost()))

    ec.run_no_migration()
    print('Total cost with no migration: {}.'.format(ec.get_cost()))

    ec.run_static()
    ec.print_migrations()
    print('Total cost for offline static algorithm: {}.'.format(ec.get_cost()))

    ec.run_belady()
