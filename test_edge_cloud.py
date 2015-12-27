#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QA test for edge cloud simulation.

from sim import EdgeCloud, parseNumRange
import pytest


class TestEdgeCloud:

    @pytest.fixture
    def ec(self):
        return EdgeCloud('traces/requests-job_id.dat', K=5, N=100)

    def test_init(self, ec):
        assert ec.K == 5 == len(ec.edge_services)
        assert len(ec.requests) == ec.N == 100
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

    def test_init_edge_services(self, ec):
        ec.init_edge_services()
        assert 0 < len(ec.edge_services) <= ec.K
        assert 0 <= ec.K_rem < ec.K
        assert sum(ec.W[s] for s in ec.edge_services) + ec.K_rem == ec.K

    def test_reset(self, ec):
        ec.reset()
        assert ec.edge_services is not None
        assert ec.cost == 0
        assert ec.migrations == []

    def test_belady_modified(self, ec):
        ec.run_belady(modified=True)
        assert ec.get_cost() == 269
        assert len(ec.migrations) == 22
        assert ec.migrations[0][0] == 4
        assert ec.migrations[-1][0] == 88
        assert ec.cost_forwarding > 0

    def test_RL(self, ec):
        ec.run_RL()
        assert ec.get_cost() == 127
        assert len(ec.migrations) == 2
        assert ec.migrations[0] == (23, 1164728954, (679429735, 754112949))
        assert ec.migrations[1] == (39, 1263655469, (757745334, 975992247))

    def test_no_migration(self, ec):
        ec.run_no_migration()
        assert ec.get_cost() == 158
        assert len(ec.migrations) == 0
        assert ec.cost == ec.cost_forwarding

    def test_static(self, ec):
        ec.run_static()
        assert ec.get_cost() >= 0
        assert len(ec.migrations) >= 0

    def test_get_cost(self, ec):
        assert ec.get_cost() == ec.cost

    def test_print_migrations(self, ec):
        pass

    def test_special(self):
        ec = EdgeCloud('traces/requests-job_id.dat', K=2, N=10,
                       special=True, M=2)
        ec.run_no_migration()
        assert ec.get_cost() == 8
        ec.run_static()
        assert ec.get_cost() == 11
        ec.run_RL()
        assert ec.get_cost() == 8
        ec.run_belady()
        assert ec.get_cost() == 14
        ec.run_belady(modified=True)
        assert ec.get_cost() == 9


def test_parseNumRange():
    assert parseNumRange('1-10') == list(range(1, 11))
    assert parseNumRange('2:5') == list(range(2, 6))
    assert parseNumRange('2.5') == [2.5]
    assert parseNumRange('5') == [5]
