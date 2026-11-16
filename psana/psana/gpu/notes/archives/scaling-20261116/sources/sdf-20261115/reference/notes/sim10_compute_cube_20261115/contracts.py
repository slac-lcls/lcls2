"""Host-only memory and callback admission contracts."""
from reference import require


def memory_demand(stage, batch):
    require(stage in ('S', 'E', 'EC', 'ECC') and batch in (50, 200), 'Unknown stage/batch')
    images = batch * (32 << 20) if stage != 'S' else 0
    if stage in ('EC', 'ECC'):
        images += batch * (64 << 20)
    return dict(framework_ceiling=6 << 30, images=images,
                cube=(100 * ((64 << 20) + 8) if stage == 'ECC' else 0),
                auxiliary=16 << 20, safety=4 << 30)


def admission(stage, batch, free, committed, allocations):
    demand = memory_demand(stage, batch)
    require(0 <= committed <= demand['framework_ceiling'] and
            sum(item['capacity'] for item in allocations) == committed, 'Framework ledger cannot be reconciled')
    outstanding = sum(demand.values()) - committed
    require(free >= outstanding, 'Capacity rejection: outstanding allocations plus safety exceed CUDA free bytes')
    return dict(demand=demand, framework_resident=committed, framework_allocations=allocations,
                outstanding=outstanding, cuda_free_at_admission=free,
                interpretation='CUDA free excludes all resident memory; subtract charged resident framework bytes once from ceiling. Task images/bank/auxiliary remain outstanding. Pool retention is included in CUDA used, never credited as free.')


def admission_from_budget(stage, batch, free, budget):
    return admission(stage, batch, free, budget.committed(), budget.allocation_snapshot())
