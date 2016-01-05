#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QA test for edge cloud simulation.

from sim import EdgeCloud, parseNumRange, get_requests
import pytest
import random


class TestEdgeCloud:

    @pytest.fixture
    def ec(self):
        requests = get_requests('traces/requests-job_id.dat', N=5000)
        return EdgeCloud(requests, K=5, M=5)

    @pytest.fixture
    def ec10(self):
        requests = get_requests('traces/requests-job_id.dat', N=10)
        return EdgeCloud(requests, K=1, M=1)

    def test_init(self, ec):
        assert ec.K == 5 == len(ec.edge_services)
        assert ec.M == 5
        assert len(ec.requests) == ec.N == 5000
        assert ec.sorted_requests[0] < ec.sorted_requests[1] \
                                     < ec.sorted_requests[-1]
        assert ec.cost == 0
        assert ec.migrations == []

    def test_reset(self, ec):
        ec.reset()
        assert ec.edge_services is not None
        assert ec.cost == 0
        assert ec.migrations == []
        assert ec.requests_seen == []

    def test_belady(self, ec):
        ec.run_belady()
        assert ec.get_cost() == 7730
        assert len(ec.migrations) == 1546
        # There is no forwarding for missing services.
        assert ec.cost_forwarding == 0
        assert ec.cost == ec.cost_migration == len(ec.migrations) * ec.M

    def test_belady_modified(self, ec10):
        ec10.run_belady(modified=True)
        assert ec10.get_cost() == 8
        assert len(ec10.migrations) == 2
        assert ec10.migrations[0][0] == 5
        assert ec10.cost_forwarding > 0

    def test_online_randomized(self, ec):
        random.seed(0)
        ec.run_online_randomized()
        assert ec.get_cost() == 4485
        assert len(ec.migrations) == 500

    def test_RL(self, ec):
        ec.run_RL()
        assert ec.get_cost() == 2566
        assert len(ec.migrations) == 56
        assert ec.cost_migration == len(ec.migrations) * ec.M
        assert ec.migrations[0] == (72, 1263655469, 42677539)
        assert ec.migrations[-1] == (4946, 1213243701, 1412105351)

    def test_no_migration(self, ec):
        ec.run_no_migration()
        assert ec.get_cost() == 4995

    def test_static(self, ec):
        ec.run_static()
        assert ec.get_cost() == 3195
        assert len(ec.migrations) == 1
        migration = ec.migrations[0]
        assert len(migration[1]) == 5 == len(migration[2])

    def test_offline_opt(self, ec10):
        ec10.run_offline_opt()
        assert ec10.get_cost() == 8
        assert len(ec10.migrations) == 5
        assert ec10.migrations[0][0] == 2

    def test_offline_iterative(self, ec10):
        ec10.run_offline_iterative()
        # Same as run_offline_opt()
        assert ec10.get_cost() == 8
        assert len(ec10.migrations) == 5
        assert ec10.migrations[0][0] == 2

    def test_get_cost(self, ec):
        assert ec.get_cost() == ec.cost

    def test_print_migrations(self, ec):
        pass

    def test_float_M(self):
        requests = get_requests('traces/requests-job_id.dat', N=1000)
        ec = EdgeCloud(requests, K=5, M=1.2)
        ec.run_RL()
        assert True


def test_get_requests():
    requests = get_requests('traces/requests-job_id.dat')
    assert len(requests) > 3e6
    N = 10000
    requests = get_requests('traces/requests-job_id.dat', N)
    assert len(requests) == N


def test_parseNumRange():
    assert parseNumRange('1-10') == list(range(1, 11))
    assert parseNumRange('2:5') == list(range(2, 6))
    assert parseNumRange('2.5') == [2.5]
    assert parseNumRange('5') == [5]
