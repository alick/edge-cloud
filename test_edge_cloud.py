#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QA test for edge cloud simulation.

from sim import EdgeCloud, parseNumRange
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
        assert len(ec.services) == ec.N_unique

    def test_gen_het_services(self, ec):
        ec.gen_het_services()
        assert len(ec.F) == ec.N_unique
        # Inspect the heterogeneity.
        F_cnt = [0] * 3
        for f in ec.F.values():
            F_cnt[f - 1] += 1
        # Ensure completeness.
        assert sum(F_cnt) == ec.N_unique
        # Roughly uniformly distributed.
        assert min(F_cnt) > 0
        assert max(F_cnt) - min(F_cnt) < ec.N_unique / 2
        # Similar for W
        assert len(ec.W) == ec.N_unique
        W_cnt = [0] * 3
        for w in ec.W.values():
            W_cnt[w - 1] += 1
        assert sum(W_cnt) == ec.N_unique
        assert min(W_cnt) > 0
        assert max(W_cnt) - min(W_cnt) < ec.N_unique / 2
        # Similar for M
        assert len(ec.M) == ec.N_unique
        M_cnt = [0] * 3
        for m in ec.M.values():
            M_cnt[m // 5 - 1] += 1
        assert sum(M_cnt) == ec.N_unique
        assert min(M_cnt) > 0
        assert max(M_cnt) - min(M_cnt) < ec.N_unique / 2

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

    def test_belady_modified(self, ec):
        ec = EdgeCloud('traces/requests-job_id.dat', K=1, M=1, N=10)
        ec.run_belady(modified=True)
        assert ec.get_cost() == 8
        assert len(ec.migrations) == 2
        assert ec.migrations[0][0] == 5
        assert ec.cost_forwarding > 0

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

    def test_offline_opt(self, ec):
        ec = EdgeCloud('traces/requests-job_id.dat', K=1, M=1, N=10)
        ec.run_offline_opt()
        assert ec.get_cost() == 8
        assert len(ec.migrations) == 5
        assert ec.migrations[0][0] == 2

    def test_offline_iterative(self, ec):
        ec = EdgeCloud('traces/requests-job_id.dat', K=1, M=1, N=10)
        ec.run_offline_iterative()
        # Same as run_offline_opt()
        assert ec.get_cost() == 8
        assert len(ec.migrations) == 5
        assert ec.migrations[0][0] == 2
        ec2 = EdgeCloud('traces/requests-job_id.dat', K=4, M=1, N=4)
        ec2.run_offline_iterative()
        assert ec2.get_cost() == 0
        assert len(ec2.migrations) == 1
        assert ec2.migrations[0][0] == 2
        ec3 = EdgeCloud('traces/requests-job_id.dat', K=2, M=1, N=15)
        ec3.run_offline_iterative()
        assert ec3.get_cost() == 12
        assert len(ec3.migrations) == 13
        assert ec3.migrations[0][0] == 2
        ec3.edge_services_matrix[0, -1] = 1390006181

    def test_get_cost(self, ec):
        assert ec.get_cost() == ec.cost

    def test_print_migrations(self, ec):
        pass

    def test_float_M(self):
        ec = EdgeCloud('traces/requests-job_id.dat', K=5, M=1.2, N=1000)
        ec.run_RL()
        assert True


def test_parseNumRange():
    assert parseNumRange('1-10') == list(range(1, 11))
    assert parseNumRange('2:5') == list(range(2, 6))
    assert parseNumRange('2.5') == [2.5]
    assert parseNumRange('5') == [5]
