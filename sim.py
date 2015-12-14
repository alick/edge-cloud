#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulation of Edge Cloud Migration

# defaultdict provides default values for missing keys
from collections import defaultdict


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

    def run_online(self):
        raise Exception('Unimplemented!')

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

    ec.run_no_migration()
    print('Total cost with no migration: {}.'.format(ec.get_cost()))

    ec.run_static()
    ec.print_migrations()
    print('Total cost for offline static algorithm: {}.'.format(ec.get_cost()))

    ec.run_online()
    ec.run_belady()
