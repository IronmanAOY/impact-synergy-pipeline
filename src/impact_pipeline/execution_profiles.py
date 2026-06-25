from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HunterSlurmProfile:
    apu_partition: str = "apu"
    cpu_partition: str = "cpu"
    account: str | None = None
    qos: str | None = None
    phase1_time: str = "24:00:00"
    cut_time: str = "24:00:00"
    reduce_time: str = "02:00:00"
    cpus_per_task: int = 32
    mem_per_task: str = "0"
    gpus_per_task: int = 0


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    description: str
    distributed_iim: bool
    hunter_phase1_shards_per_run: int = 1
    hunter_cut_shards_per_run: int = 1
    hunter_phase1_workers_per_task: int | None = None
    hunter_phase1_chunk_size: int = 8
    hunter_shared_memory: bool = True
    hunter_slurm: HunterSlurmProfile | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


LOCAL_EXECUTION_PROFILE = ExecutionProfile(
    name="local",
    description="Single-machine execution for workstation development and stepwise runs.",
    distributed_iim=False,
    hunter_phase1_shards_per_run=1,
    hunter_cut_shards_per_run=1,
    hunter_phase1_workers_per_task=None,
    hunter_phase1_chunk_size=8,
    hunter_shared_memory=True,
    hunter_slurm=None,
    notes=(
        "Uses the in-process pipeline and local multiprocessing only.",
        "Intended for MacBook-scale debugging, smoke tests, and individual step execution.",
    ),
)


HUNTER_EXECUTION_PROFILE = ExecutionProfile(
    name="hunter",
    description="Hunter-oriented distributed IIM execution with shared math and Slurm shard orchestration.",
    distributed_iim=True,
    hunter_phase1_shards_per_run=16,
    hunter_cut_shards_per_run=256,
    hunter_phase1_workers_per_task=32,
    hunter_phase1_chunk_size=8,
    hunter_shared_memory=True,
    hunter_slurm=HunterSlurmProfile(),
    notes=(
        "IIM is decomposed into prepare, phase-1, cut-shard, and reduce stages.",
        "Non-IIM pipeline stages remain shared with local mode to avoid double maintenance.",
        "Slurm partition names and account/QoS can be overridden at runtime.",
    ),
)


def get_execution_profile(name: str) -> ExecutionProfile:
    key = str(name).strip().lower()
    if key == "local":
        return LOCAL_EXECUTION_PROFILE
    if key == "hunter":
        return HUNTER_EXECUTION_PROFILE
    raise ValueError(f"Unknown execution mode '{name}'. Expected one of: local, hunter.")
