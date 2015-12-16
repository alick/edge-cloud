#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QA test for edge cloud simulation.

from sim import EdgeCloud
import pytest


class TestEdgeCloud:

    @pytest.fixture
    def ec(self):
        return EdgeCloud('traces/requests-job_id.dat', K=5, M=5, N=5000)

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
        pass

    def test_RL(self, ec):
        ec.run_RL()
        assert ec.get_cost() == 2566
        assert len(ec.migrations) == 56
        assert ec.cost_migration == len(ec.migrations) * ec.M
        assert ec.migrations[0] == (72, 1263655469, 143787193)
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

    def test_get_cost(self, ec):
        assert ec.get_cost() == ec.cost

    def test_print_migrations(self, ec):
        pass

    def test_float_M(self):
        ec = EdgeCloud('traces/requests-job_id.dat', K=5, M=1.2, N=1000)
        ec.run_RL()
        assert True
