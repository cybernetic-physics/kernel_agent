#!POPCORN leaderboard nvfp4_group_gemm
#!POPCORN gpu NVIDIA
#!AFTERHOURS harness tools/pygpubench_nvfp4_dual_gemm_harness.py
# NOTE: Derived from wagmiv67.py with 192-thread warp specialization and
# persistent grouped scheduling / ACC-stage overlap.
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

import functools
import gc
from typing import Tuple, List

import torch
from task import input_t, output_t

gc.disable()

# Kernel configuration parameters
# Size of tma descriptor in bytes
bytes_per_tensormap = 128
# Number of tensormaps: a, b, sfa, sfb
num_tensormaps = 4
# Tile sizes for M, N, K dimensions
mma_tiler_mnk = (128, 128, 256)
# Shape of the K dimension for the MMA instruction
mma_inst_shape_k = 64
# FP4 data type for A and B
ab_dtype = cutlass.Float4E2M1FN  
# FP8 data type for scale factors
sf_dtype = cutlass.Float8E4M3FN  
# FP16 output type
c_dtype = cutlass.Float16  
# Scale factor block size (16 elements share one scale)
sf_vec_size = 16  
# Number of threads per CUDA thread block
threads_per_cta = 192
epilogue_warp_count = 4
mma_warp_id = 4
tma_warp_id = 5
# Stage numbers of shared memory and tmem
num_acc_stage = 2
num_ab_stage = 6
persistent_wave_multiplier = 1


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


def round_tmem_alloc_cols(required_cols: int) -> int:
    """
    TMEM allocator accepts power-of-two column counts that are multiples of 32.
    Valid values are {32, 64, 128, 256, 512}.
    """
    valid_cols = (32, 64, 128, 256, 512)
    need = max(1, int(required_cols))
    for cols in valid_cols:
        if need <= cols:
            return cols
    # Keep previous behavior upper bound when required footprint exceeds valid set.
    return 512


# The CuTe reference implementation for NVFP4 block-scaled GEMM
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    tensor_of_abc_ptrs: cute.Tensor,
    tensor_of_sfasfb_ptrs: cute.Tensor,
    tensormaps: cute.Tensor,
    tensor_of_problem_sizes: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    tensor_of_cta_prefix: cute.Tensor,
    num_groups: cutlass.Constexpr[int],
    num_tma_load_bytes: cutlass.Constexpr[int],
):
    """
    GPU device kernel performing the Group GEMM computation.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()
    is_epilogue_warp = warp_idx < epilogue_warp_count
    is_mma_warp = warp_idx == mma_warp_id
    is_tma_warp = warp_idx == tma_warp_id

    #
    # Persistent grouped scheduler.
    #
    bidx, _, _ = cute.arch.block_idx()
    grid_dim_x, _, _ = cute.arch.grid_dim()
    total_tiles = tensor_of_cta_prefix[num_groups]
    tiles_per_cta = ceil_div(total_tiles, grid_dim_x)
    tile_start = bidx * tiles_per_cta
    tile_end = tile_start + tiles_per_cta
    if tile_end > total_tiles:
        tile_end = total_tiles

    #
    # Define shared storage for kernel
    #
    size_tensormap_in_i64 = (
        num_tensormaps * bytes_per_tensormap // 8
    )
    @cute.struct
    class SharedStorage:
        tensormap_buffer: cute.struct.MemRange[
            cutlass.Int64, size_tensormap_in_i64
        ]
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_holding_buf: cutlass.Int32
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
    tensormap_a_smem_ptr = tensormap_smem_ptr
    tensormap_b_smem_ptr = (
        tensormap_a_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfa_smem_ptr = (
        tensormap_b_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfb_smem_ptr = (
        tensormap_sfa_smem_ptr
        + bytes_per_tensormap // 8
    )
    # Setup smem tensor for A, B, SFA, SFB
    # (MMA, MMA_M, MMA_K, STAGE)
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    # (MMA, MMA_M, MMA_K, STAGE)
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )
    tile_coord_x_smem = smem.allocate_tensor(
        element_type=cutlass.Int32,
        layout=cute.make_layout((num_acc_stage), stride=(1)),
        byte_alignment=16,
    )
    tile_coord_y_smem = smem.allocate_tensor(
        element_type=cutlass.Int32,
        layout=cute.make_layout((num_acc_stage), stride=(1)),
        byte_alignment=16,
    )
    tile_m_smem = smem.allocate_tensor(
        element_type=cutlass.Int32,
        layout=cute.make_layout((num_acc_stage), stride=(1)),
        byte_alignment=16,
    )
    tile_n_smem = smem.allocate_tensor(
        element_type=cutlass.Int32,
        layout=cute.make_layout((num_acc_stage), stride=(1)),
        byte_alignment=16,
    )
    tile_l_smem = smem.allocate_tensor(
        element_type=cutlass.Int32,
        layout=cute.make_layout((num_acc_stage), stride=(1)),
        byte_alignment=16,
    )
    tile_c_ptr_smem = smem.allocate_tensor(
        element_type=cutlass.Int64,
        layout=cute.make_layout((num_acc_stage), stride=(1)),
        byte_alignment=16,
    )

    # Initialize mainloop ab_pipeline, acc_pipeline and their states
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            epilogue_warp_count * 32,
        ),
    ).make_participants()

    #
    # Local_tile partition global tensors
    #
    # (bM, bK, RestM, RestK, RestL)
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bM, bK, RestM, RestK, RestL)
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    #
    # Partition global tensor for TiledMMA_A/B/C
    #
    # The MMA partition domain is 128 threads. For 192-thread CTAs, remap the
    # extra two warps into the valid 0..127 slice range.
    mma_part_slice_idx = tidx
    if mma_part_slice_idx >= epilogue_warp_count * 32:
        mma_part_slice_idx = mma_part_slice_idx - epilogue_warp_count * 32
    thr_mma = tiled_mma.get_slice(mma_part_slice_idx)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgA = thr_mma.partition_A(gA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB = thr_mma.partition_B(gB_nkl)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB = thr_mma.partition_B(gSFB_nkl)
    # Update tma descriptor with the correct shapes and strides
    tensormap_manager = utils.TensorMapManager(
        utils.TensorMapUpdateMode.GMEM,
        128,
    )
    # Use one descriptor workspace per CTA (indexed by blockIdx.x) and update it
    # as each persistent tile is assigned to this CTA.
    tensormap_workspace_idx = bidx
    tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(tensormap_workspace_idx, 0, None)].iterator
    )
    tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(tensormap_workspace_idx, 1, None)].iterator
    )
    tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(tensormap_workspace_idx, 2, None)].iterator
    )
    tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(tensormap_workspace_idx, 3, None)].iterator
    )
    tensormap_init_barrier = pipeline.NamedBarrier(
        barrier_id=2,
        num_threads=64,
    )

    # Match reference initialization flow: one warp initializes SMEM descriptors,
    # then TMA warp performs dynamic updates for persistent tiles.
    if is_tma_warp or is_mma_warp:
        if is_mma_warp:
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_a, tensormap_a_gmem_ptr, mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_gmem_ptr, mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfa, tensormap_sfa_gmem_ptr, mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfb, tensormap_sfb_gmem_ptr, mma_warp_id
            )
        tensormap_init_barrier.arrive_and_wait()

    #
    # Partition global/shared tensor for TMA load A/B/SFA/SFB
    #
    # TMA Partition_S/D for A
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # TMA Partition_S/D for B
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    #  TMA Partition_S/D for SFA
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    # TMA Partition_S/D for SFB
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    #
    # Partition shared/tensor memory tensor for TiledMMA_A/B/C
    #
    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    # Build SFA/SFB TMEM layouts before allocation so footprint can be computed.
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )

    tCtSFA_fake = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
        tCtSFA_layout,
    )
    tCtSFB_fake = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
        tCtSFB_layout,
    )
    acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAcc_fake)
    sfa_cols = tcgen05.find_tmem_tensor_col_offset(tCtSFA_fake)
    sfb_cols = tcgen05.find_tmem_tensor_col_offset(tCtSFB_fake)
    total_tmem_cols = acc_cols * num_acc_stage + sfa_cols + sfb_cols
    alloc_tmem_cols = round_tmem_alloc_cols(total_tmem_cols)

    #
    # Alloc tensor memory buffer
    #
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta - 32,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    tmem.allocate(alloc_tmem_cols)
    if not is_tma_warp:
        tmem.wait_for_alloc()
    acc_tmem_base_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc_stage0 = cute.make_tensor(acc_tmem_base_ptr, tCtAcc_fake.layout)

    #
    # Make SFA/SFB tmem tensor
    #
    # Get SFA tmem ptr
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_base_ptr + acc_cols * num_acc_stage,
        dtype=sf_dtype,
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
    # Get SFB tmem ptr
    sfb_tmem_ptr = cute.recast_ptr(
        acc_tmem_base_ptr
        + acc_cols * num_acc_stage
        + sfa_cols,
        dtype=sf_dtype,
    )
    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

    #
    # Partition for S2T copy of SFA/SFB
    #
    # Make S2T CopyAtom
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact = cute.filter_zeros(sSFB)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB_compact = cute.filter_zeros(tCtSFB)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

    num_kblocks = cute.size(tCrA, mode=[2])

    #
    # Persistent producer loop (TMA warp)
    #
    if is_tma_warp and tile_start < tile_end:
        tile_idx = tile_start
        prev_group_idx = cutlass.Int32(-1)
        cta_m = cutlass.Int32(0)
        k_tile_cnt = cutlass.Int32(0)
        coord_x = cutlass.Int32(0)
        coord_y = cutlass.Int32(0)
        m = cutlass.Int32(0)
        n = cutlass.Int32(0)
        l = cutlass.Int32(0)
        c_ptr = cutlass.Int64(0)
        group_idx = cutlass.Int32(0)
        if cutlass.const_expr(num_groups == 2):
            p1 = tensor_of_cta_prefix[1]
            if tile_idx >= p1:
                group_idx = cutlass.Int32(1)
        elif cutlass.const_expr(num_groups == 8):
            p1 = tensor_of_cta_prefix[1]
            p2 = tensor_of_cta_prefix[2]
            p3 = tensor_of_cta_prefix[3]
            p4 = tensor_of_cta_prefix[4]
            p5 = tensor_of_cta_prefix[5]
            p6 = tensor_of_cta_prefix[6]
            p7 = tensor_of_cta_prefix[7]
            if tile_idx < p4:
                if tile_idx < p2:
                    if tile_idx < p1:
                        group_idx = cutlass.Int32(0)
                    else:
                        group_idx = cutlass.Int32(1)
                else:
                    if tile_idx < p3:
                        group_idx = cutlass.Int32(2)
                    else:
                        group_idx = cutlass.Int32(3)
            else:
                if tile_idx < p6:
                    if tile_idx < p5:
                        group_idx = cutlass.Int32(4)
                    else:
                        group_idx = cutlass.Int32(5)
                else:
                    if tile_idx < p7:
                        group_idx = cutlass.Int32(6)
                    else:
                        group_idx = cutlass.Int32(7)
        else:
            left = cutlass.Int32(0)
            right = num_groups
            while left < right:
                mid = (left + right) // 2
                if tensor_of_cta_prefix[mid + 1] <= tile_idx:
                    left = mid + 1
                else:
                    right = mid
            group_idx = left
        group_end = tensor_of_cta_prefix[group_idx + 1]
        tensormap_a_desc_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_a_gmem_ptr,
            cute.AddressSpace.generic,
        )
        tensormap_b_desc_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_b_gmem_ptr,
            cute.AddressSpace.generic,
        )
        tensormap_sfa_desc_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_sfa_gmem_ptr,
            cute.AddressSpace.generic,
        )
        tensormap_sfb_desc_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_sfb_gmem_ptr,
            cute.AddressSpace.generic,
        )
        while tile_idx < tile_end:
            if tile_idx >= group_end:
                group_idx = group_idx + 1
                group_end = tensor_of_cta_prefix[group_idx + 1]
            if group_idx != prev_group_idx:
                m = tensor_of_problem_sizes[group_idx, 0]
                n = tensor_of_problem_sizes[group_idx, 1]
                k = tensor_of_problem_sizes[group_idx, 2]
                l = tensor_of_problem_sizes[group_idx, 3]
                cta_m = ceil_div(m, mma_tiler_mnk[0])
                k_tile_cnt = cute.ceil_div(k, mma_tiler_mnk[2])
                cta_rest = tile_idx - tensor_of_cta_prefix[group_idx]
                coord_y = cta_rest // cta_m
                coord_x = cta_rest - coord_y * cta_m

                mA_mkl_iter = cute.make_ptr(
                    ab_dtype, tensor_of_abc_ptrs[group_idx, 0], cute.AddressSpace.gmem
                ).align(32)
                mB_nkl_iter = cute.make_ptr(
                    ab_dtype, tensor_of_abc_ptrs[group_idx, 1], cute.AddressSpace.gmem
                ).align(32)
                sfa_mkl_iter = cute.make_ptr(
                    sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 0], cute.AddressSpace.gmem
                ).align(32)
                sfb_nkl_iter = cute.make_ptr(
                    sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 1], cute.AddressSpace.gmem
                ).align(32)
                mA_mkl_layout = cute.make_layout(
                    (m, k, l), stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32),))
                mB_nkl_layout = cute.make_layout(
                    (n, k, l), stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32),))
                sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                    mA_mkl_layout.shape, sf_vec_size
                )
                sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                    mB_nkl_layout.shape, sf_vec_size
                )
                real_tensor_a = cute.make_tensor(mA_mkl_iter, mA_mkl_layout)
                real_tensor_b = cute.make_tensor(mB_nkl_iter, mB_nkl_layout)
                real_tensor_sfa = cute.make_tensor(sfa_mkl_iter, sfa_layout)
                real_tensor_sfb = cute.make_tensor(sfb_nkl_iter, sfb_layout)

                tensormap_manager.update_tensormap(
                    (
                        real_tensor_a,
                        real_tensor_b,
                        real_tensor_sfa,
                        real_tensor_sfb,
                    ),
                    (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
                    (
                        tensormap_a_gmem_ptr,
                        tensormap_b_gmem_ptr,
                        tensormap_sfa_gmem_ptr,
                        tensormap_sfb_gmem_ptr,
                    ),
                    tma_warp_id,
                    (
                        tensormap_a_smem_ptr,
                        tensormap_b_smem_ptr,
                        tensormap_sfa_smem_ptr,
                        tensormap_sfb_smem_ptr,
                    ),
                )
                tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)
                prev_group_idx = group_idx

            tAgA_tile = tAgA[(None, coord_x, None, 0)]
            tBgB_tile = tBgB[(None, coord_y, None, 0)]
            tAgSFA_tile = tAgSFA[(None, coord_x, None, 0)]
            tBgSFB_tile = tBgSFB[(None, coord_y, None, 0)]
            for k_tile in range(k_tile_cnt):
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_tile[(None, k_tile)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=tensormap_a_desc_ptr,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_tile[(None, k_tile)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=tensormap_b_desc_ptr,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA_tile[(None, k_tile)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=tensormap_sfa_desc_ptr,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB_tile[(None, k_tile)],
                    tBsSFB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=tensormap_sfb_desc_ptr,
                )
            coord_x = coord_x + 1
            if coord_x == cta_m:
                coord_x = cutlass.Int32(0)
                coord_y = coord_y + 1
            tile_idx += 1

    #
    # Persistent consumer loop (MMA warp)
    #
    if is_mma_warp and tile_start < tile_end:
        tile_idx = tile_start
        prev_group_idx = cutlass.Int32(-1)
        cta_m = cutlass.Int32(0)
        k_tile_cnt = cutlass.Int32(0)
        coord_x = cutlass.Int32(0)
        coord_y = cutlass.Int32(0)
        m = cutlass.Int32(0)
        n = cutlass.Int32(0)
        l = cutlass.Int32(0)
        c_ptr = cutlass.Int64(0)
        group_idx = cutlass.Int32(0)
        if cutlass.const_expr(num_groups == 2):
            p1 = tensor_of_cta_prefix[1]
            if tile_idx >= p1:
                group_idx = cutlass.Int32(1)
        elif cutlass.const_expr(num_groups == 8):
            p1 = tensor_of_cta_prefix[1]
            p2 = tensor_of_cta_prefix[2]
            p3 = tensor_of_cta_prefix[3]
            p4 = tensor_of_cta_prefix[4]
            p5 = tensor_of_cta_prefix[5]
            p6 = tensor_of_cta_prefix[6]
            p7 = tensor_of_cta_prefix[7]
            if tile_idx < p4:
                if tile_idx < p2:
                    if tile_idx < p1:
                        group_idx = cutlass.Int32(0)
                    else:
                        group_idx = cutlass.Int32(1)
                else:
                    if tile_idx < p3:
                        group_idx = cutlass.Int32(2)
                    else:
                        group_idx = cutlass.Int32(3)
            else:
                if tile_idx < p6:
                    if tile_idx < p5:
                        group_idx = cutlass.Int32(4)
                    else:
                        group_idx = cutlass.Int32(5)
                else:
                    if tile_idx < p7:
                        group_idx = cutlass.Int32(6)
                    else:
                        group_idx = cutlass.Int32(7)
        else:
            left = cutlass.Int32(0)
            right = num_groups
            while left < right:
                mid = (left + right) // 2
                if tensor_of_cta_prefix[mid + 1] <= tile_idx:
                    left = mid + 1
                else:
                    right = mid
            group_idx = left
        group_end = tensor_of_cta_prefix[group_idx + 1]
        while tile_idx < tile_end:
            if tile_idx >= group_end:
                group_idx = group_idx + 1
                group_end = tensor_of_cta_prefix[group_idx + 1]
            if group_idx != prev_group_idx:
                m = tensor_of_problem_sizes[group_idx, 0]
                n = tensor_of_problem_sizes[group_idx, 1]
                k = tensor_of_problem_sizes[group_idx, 2]
                l = tensor_of_problem_sizes[group_idx, 3]
                c_ptr = tensor_of_abc_ptrs[group_idx, 2]
                cta_m = ceil_div(m, mma_tiler_mnk[0])
                k_tile_cnt = cute.ceil_div(k, mma_tiler_mnk[2])
                cta_rest = tile_idx - tensor_of_cta_prefix[group_idx]
                coord_y = cta_rest // cta_m
                coord_x = cta_rest - coord_y * cta_m
                prev_group_idx = group_idx

            acc_empty = acc_producer.acquire_and_advance()
            stage_idx = acc_empty.index
            tile_coord_x_smem[stage_idx] = coord_x
            tile_coord_y_smem[stage_idx] = coord_y
            tile_m_smem[stage_idx] = m
            tile_n_smem[stage_idx] = n
            tile_l_smem[stage_idx] = l
            tile_c_ptr_smem[stage_idx] = c_ptr
            if cutlass.const_expr(num_acc_stage == 1):
                tCtAcc = tCtAcc_stage0
            else:
                tCtAcc = cute.make_tensor(
                    acc_tmem_base_ptr + stage_idx * acc_cols,
                    tCtAcc_fake.layout,
                )

            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            accumulate_enabled = False
            for k_tile in range(k_tile_cnt):
                ab_full = ab_consumer.wait_and_advance()
                s2t_stage_coord = (None, None, None, None, ab_full.index)
                tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t_staged,
                    tCtSFA_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t_staged,
                    tCtSFB_compact_s2t,
                )

                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    kblock_coord = (
                        None,
                        None,
                        kblock_idx,
                        ab_full.index,
                    )
                    sf_kblock_coord = (None, None, kblock_idx)
                    tiled_mma.set(
                        tcgen05.Field.SFA,
                        tCtSFA[sf_kblock_coord].iterator,
                    )
                    tiled_mma.set(
                        tcgen05.Field.SFB,
                        tCtSFB[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[kblock_coord],
                        tCrB[kblock_coord],
                        tCtAcc,
                    )
                    if not accumulate_enabled:
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        accumulate_enabled = True
                ab_full.release()
            acc_empty.commit()
            coord_x = coord_x + 1
            if coord_x == cta_m:
                coord_x = cutlass.Int32(0)
                coord_y = coord_y + 1
            tile_idx += 1

    #
    # Persistent epilogue loop (epilogue warps)
    #
    op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
    copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
    epilogue_slice_idx = tidx
    if not is_epilogue_warp:
        epilogue_slice_idx = 0
    simt_atom_128 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=128
    )
    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=16
    )
    thread_row = tidx
    if cutlass.const_expr(num_acc_stage == 1):
        tiled_copy_t2r_stage0 = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc_stage0[None, 0, 0]
        )
        thr_copy_t2r_stage0 = tiled_copy_t2r_stage0.get_slice(epilogue_slice_idx)
        tDtAcc_stage0 = thr_copy_t2r_stage0.partition_S(tCtAcc_stage0[None, 0, 0])

    if is_epilogue_warp and tile_start < tile_end:
        tile_idx = tile_start
        m = cutlass.Int32(0)
        n = cutlass.Int32(0)
        l = cutlass.Int32(0)
        c_ptr = cutlass.Int64(0)
        while tile_idx < tile_end:
            acc_full = acc_consumer.wait_and_advance()
            stage_idx = acc_full.index
            coord_x = tile_coord_x_smem[stage_idx]
            coord_y = tile_coord_y_smem[stage_idx]
            m = tile_m_smem[stage_idx]
            n = tile_n_smem[stage_idx]
            l = tile_l_smem[stage_idx]
            c_ptr = tile_c_ptr_smem[stage_idx]

            if cutlass.const_expr(num_acc_stage == 1):
                tiled_copy_t2r = tiled_copy_t2r_stage0
                thr_copy_t2r = thr_copy_t2r_stage0
                tDtAcc = tDtAcc_stage0
            else:
                tCtAcc = cute.make_tensor(
                    acc_tmem_base_ptr + stage_idx * acc_cols,
                    tCtAcc_fake.layout,
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[None,0,0])
                thr_copy_t2r = tiled_copy_t2r.get_slice(epilogue_slice_idx)
                tDtAcc = thr_copy_t2r.partition_S(tCtAcc[None,0,0])

            mC_mnl_iter = cute.make_ptr(
                c_dtype, c_ptr, cute.AddressSpace.gmem
            ).align(32)
            mC_mnl_layout = cute.make_layout(
                (m, n, l),
                stride=(cute.assume(n, 32), 1, cute.assume(m * n, 32),))
            mC_mnl = cute.make_tensor(mC_mnl_iter, mC_mnl_layout)
            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (coord_x, coord_y, 0)
            )
            tCgC = thr_mma.partition_C(gC_mnl)
            tDgC = thr_copy_t2r.partition_D(tCgC[None,0,0])
            tDrAcc = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
            tDrC = cute.make_rmem_tensor(tDgC.shape, c_dtype)

            residue_m = mC_mnl.shape[0] - cutlass.Int32(coord_x) * mma_tiler_mnk[0]
            residue_n = mC_mnl.shape[1] - cutlass.Int32(coord_y) * mma_tiler_mnk[1]
            full_m_tile = residue_m >= mma_tiler_mnk[0]
            full_n_tile = residue_n >= mma_tiler_mnk[1]
            row_valid = thread_row < residue_m
            has_output_row = full_m_tile or row_valid

            cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
            if has_output_row:
                tDrC.store(tDrAcc.load().to(c_dtype))

            if has_output_row and full_n_tile:
                cute.copy(simt_atom_128, cute.flatten(tDrC), cute.flatten(tDgC))
            elif has_output_row:
                tDpC = cute.make_rmem_tensor(tDrC.shape, cutlass.Boolean)
                for i in cutlass.range(cute.size(tDrC.shape), unroll_full=True):
                    tDpC[i] = i < residue_n
                cute.copy(
                    simt_atom,
                    cute.flatten(tDrC),
                    cute.flatten(tDgC),
                    pred=cute.flatten(tDpC),
                )
            acc_full.release()
            tile_idx += 1

    tmem.relinquish_alloc_permit()
    # Deallocate TMEM
    cute.arch.barrier()
    tmem.free(acc_tmem_base_ptr)
    pass


# Host-side JIT function to prepare tensors and launch GPU kernel.
@cute.jit
def my_kernel(
    ptr_of_tensor_of_problem_sizes: cute.Pointer,
    ptr_of_tensor_of_abc_ptrs: cute.Pointer,
    ptr_of_tensor_of_sfasfb_ptrs: cute.Pointer,
    ptr_of_tensor_of_cta_prefix: cute.Pointer,
    ptr_of_tensor_of_tensormap: cute.Pointer,
    total_num_clusters: cutlass.Int32,
    persistent_blocks: cutlass.Int32,
    problem_sizes: List[
        Tuple[int, int, int, int]
    ],  # Problem sizes for each group
    num_groups: cutlass.Constexpr[int],
):
    tensor_of_abc_ptrs = cute.make_tensor(
        ptr_of_tensor_of_abc_ptrs, cute.make_layout((num_groups, 3), stride=(3, 1))
    )
    tensor_of_sfasfb_ptrs = cute.make_tensor(
        ptr_of_tensor_of_sfasfb_ptrs, cute.make_layout((num_groups, 2), stride=(2, 1))
    )
    tensor_of_problem_sizes = cute.make_tensor(
        ptr_of_tensor_of_problem_sizes, cute.make_layout((num_groups, 4), stride=(4, 1))
    )
    tensor_of_cta_prefix = cute.make_tensor(
        ptr_of_tensor_of_cta_prefix, cute.make_layout((num_groups + 1), stride=(1))
    )
    tensor_of_tensormap = cute.make_tensor(
        ptr_of_tensor_of_tensormap, cute.make_layout((persistent_blocks, 4, 16), stride=(64, 16, 1))
    )

    # Use fake shape for initial Tma descriptor and atom setup
    # The real Tma desc and atom will be updated during kernel execution.
    min_a_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    min_b_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    initial_a = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_a_shape[0], cute.assume(min_a_shape[2], 32), min_a_shape[3]),
            stride=(
                cute.assume(min_a_shape[2], 32),
                1,
                cute.assume(min_a_shape[0] * min_a_shape[2], 32),
            ),
        ),
    )
    initial_b = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_b_shape[1], cute.assume(min_b_shape[2], 32), min_b_shape[3]),
            stride=(
                cute.assume(min_b_shape[2], 32),
                1,
                cute.assume(min_b_shape[1] * min_b_shape[2], 32),
            ),
        ),
    )

    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_a.shape, sf_vec_size
    )
    # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_b.shape, sf_vec_size
    )
    # Create initial SFA and SFB tensors with fake shape and null pointer.
    initial_sfa = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfa_layout)
    initial_sfb = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfb_layout)

    # Select MMA operation
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)),
        (tiled_mma.thr_id.shape,),
    )

    # Compute A/B/SFA/SFB/C shared memory layout
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )

    # Setup TMA for A
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_a,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for B
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_b,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for SFA
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfa,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfb,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
    )

    # Persistent grouped launch: fewer CTAs than total tiles, each CTA loops tiles.
    grid = (persistent_blocks, 1, 1)

    # Launch the kernel
    kernel(
        # MMA (Matrix Multiply-Accumulate) configuration
        tiled_mma,                  # Tiled MMA object defining NVFP4 GEMM compute pattern
        
        # TMA (Tensor Memory Accelerator) atoms and tensors for input matrix A
        tma_atom_a,                 # TMA copy atom defining how to load A from global memory
        tma_tensor_a,               # Tensor descriptor for A (created from smallest A tensor)
        
        # TMA atoms and tensors for input matrix B
        tma_atom_b,                 # TMA copy atom defining how to load B from global memory
        tma_tensor_b,               # Tensor descriptor for B (created from smallest B tensor)
        
        # TMA atoms and tensors for scale factor A
        tma_atom_sfa,               # TMA copy atom for loading scale factors for A
        tma_tensor_sfa,             # Tensor descriptor for SFA (block scale factors for A)
        
        # TMA atoms and tensors for scale factor B
        tma_atom_sfb,               # TMA copy atom for loading scale factors for B
        tma_tensor_sfb,             # Tensor descriptor for SFB (block scale factors for B)
        
        # Runtime tensor metadata for dynamic group access
        tensor_of_abc_ptrs,         # Device tensor containing pointers to A, B, C for all groups
        tensor_of_sfasfb_ptrs,      # Device tensor containing pointers to SFA, SFB for all groups
        tensor_of_tensormap,        # Pre-allocated buffer for tensormap descriptors per CTA
        tensor_of_problem_sizes,    # Device tensor containing (m, n, k, l) for each group
        
        # Shared memory layouts with staging for pipelined execution
        a_smem_layout_staged,       # Staged shared memory layout for A (includes stage dimension)
        b_smem_layout_staged,       # Staged shared memory layout for B (includes stage dimension)
        sfa_smem_layout_staged,     # Staged shared memory layout for SFA (includes stage dimension)
        sfb_smem_layout_staged,     # Staged shared memory layout for SFB (includes stage dimension)
        
        # CTA grid configuration
        tensor_of_cta_prefix,       # Prefix sums over per-group CTA counts
        num_groups,                 # Number of groups in this batch

        # Pipeline synchronization parameter
        num_tma_load_bytes,         # Total bytes to load per TMA transaction (for barrier setup)
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
        min_blocks_per_mp=0,
    )
    return


# Global cache for compiled kernels (keyed by group size)
_compiled_kernel_cache = {}
# Runtime metadata cache keyed by exact problem-size tuples.
_runtime_meta_cache = {}
# This function is used to compile the kernel once and cache it and then allow users to 
# run the kernel multiple times to get more accurate timing results.
def compile_kernel(problem_sizes, selected_acc_stage: int | None = None):
    """
    Compile the kernel once and cache it using problem_sizes as the key.
    This should be called before any timing measurements.

    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache
    global num_acc_stage

    if selected_acc_stage is None:
        selected_acc_stage = int(num_acc_stage)
    else:
        selected_acc_stage = int(selected_acc_stage)
    
    # Cache per exact grouped shape set; len-only caching can alias incompatible specializations.
    cache_key = (
        tuple(tuple(int(x) for x in mnkl) for mnkl in problem_sizes),
        selected_acc_stage,
    )

    # Check if we already have a compiled kernel for these problem sizes
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    cute_ptr_of_tensor_of_cta_prefix = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    # Fake cluster numbers for compile only.
    total_num_clusters = cutlass.Int32(1)
    persistent_blocks = cutlass.Int32(1)
    num_groups = len(problem_sizes)
    # Each cluster needs its own set of tensormaps (one for A, B, SFA, SFB)
    # Shape: (total_num_clusters, num_tensormaps=4, bytes_per_tensormap/8=16)
    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    prev_num_acc_stage = num_acc_stage
    num_acc_stage = selected_acc_stage
    try:
        compiled_func = cute.compile(
            my_kernel,
            cute_ptr_of_tensor_of_problem_sizes,
            cute_ptr_of_tensor_of_abc_ptrs,
            cute_ptr_of_tensor_of_sfasfb_ptrs,
            cute_ptr_of_tensor_of_cta_prefix,
            cute_ptr_of_tensor_of_tensormap,
            total_num_clusters,
            persistent_blocks,
            problem_sizes,
            num_groups,
        )
    finally:
        num_acc_stage = prev_num_acc_stage
    # Store compiled kernel in cache with problem_sizes as key
    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled group GEMM kernel.
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (abc_tensors, sfasfb_tensors, problem_sizes) where:
            abc_tensors: list of tuples (a, b, c) where 
                a is torch.Tensor[float4e2m1fn_x2] of shape [m, k // 2, l]
                b is torch.Tensor[float4e2m1fn_x2] of shape [n, k // 2, l]
                c is torch.Tensor[float16] of shape [m, n, l]
            sfasfb_tensors: list of tuples (sfa, sfb) where 
                sfa is torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l]
                sfb is torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l]
            problem_sizes: list of tuples (m, n, k, l)
            each group has its own a, b, c, sfa, sfb with different m, n, k, l problem sizes
            l should always be 1 for each group.
            list size is the number of groups.
    
    Returns:
        list of c tensors where c is torch.Tensor[float16] of shape [m, n, l] for each group
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    output_abc_tensors = abc_tensors

    num_groups_input = len(problem_sizes)
    if num_groups_input > 1:
        cta_m, cta_n, cta_k = mma_tiler_mnk

        def _group_work(group_idx: int) -> int:
            m, n, k, _ = problem_sizes[group_idx]
            return (
                ceil_div(int(m), cta_m)
                * ceil_div(int(n), cta_n)
                * ceil_div(int(k), cta_k)
            )

        group_order = sorted(
            range(num_groups_input),
            key=_group_work,
            reverse=True,
        )
        if any(group_order[i] != i for i in range(num_groups_input)):
            abc_tensors = [abc_tensors[i] for i in group_order]
            sfasfb_reordered_tensors = [sfasfb_reordered_tensors[i] for i in group_order]
            problem_sizes = [problem_sizes[i] for i in group_order]

    global _runtime_meta_cache
    selected_acc_stage = 2
    compiled_func = compile_kernel(problem_sizes, selected_acc_stage)

    # Cache shape-derived launch metadata for repeated benchmark invocations.
    runtime_key = tuple(tuple(int(x) for x in mnkl) for mnkl in problem_sizes)
    runtime_meta = _runtime_meta_cache.get(runtime_key)
    runtime_meta_is_new = runtime_meta is None
    if runtime_meta is None:
        tensor_of_problem_sizes = torch.tensor(
            problem_sizes, dtype=torch.int32, device="cuda"
        )

        cta_tile_shape_mn = [mma_tiler_mnk[0], mma_tiler_mnk[1]]
        cluster_tile_shape_mn = tuple(
            x * y for x, y in zip(cta_tile_shape_mn, (1, 1))
        )

        total_num_clusters = 0
        cta_prefix = [0]
        for m, n, _, _ in problem_sizes:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
            )
            group_clusters = functools.reduce(lambda x, y: x * y, num_clusters_mn)
            total_num_clusters += group_clusters
            cta_prefix.append(total_num_clusters)
        tensor_of_cta_prefix = torch.tensor(cta_prefix, dtype=torch.int32, device="cuda")
        persistent_blocks = min(
            total_num_clusters,
            max(
                1,
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count
                * persistent_wave_multiplier,
            ),
        )

        tensormap_shape = (
            persistent_blocks,
            num_tensormaps,
            bytes_per_tensormap // 8,
        )
        tensor_of_tensormap = torch.empty(tensormap_shape, dtype=torch.int64, device="cuda")
        num_groups_local = len(problem_sizes)
        tensor_of_abc_ptrs = torch.empty((num_groups_local, 3), dtype=torch.int64, device="cuda")
        tensor_of_sfasfb_ptrs = torch.empty((num_groups_local, 2), dtype=torch.int64, device="cuda")
        host_abc_ptrs = torch.empty((num_groups_local, 3), dtype=torch.int64)
        host_sfasfb_ptrs = torch.empty((num_groups_local, 2), dtype=torch.int64)
        runtime_meta = {
            "tensor_of_problem_sizes": tensor_of_problem_sizes,
            "tensor_of_cta_prefix": tensor_of_cta_prefix,
            "tensor_of_tensormap": tensor_of_tensormap,
            "tensor_of_abc_ptrs": tensor_of_abc_ptrs,
            "tensor_of_sfasfb_ptrs": tensor_of_sfasfb_ptrs,
            "host_abc_ptrs": host_abc_ptrs,
            "host_sfasfb_ptrs": host_sfasfb_ptrs,
            "cute_ptr_of_tensor_of_abc_ptrs": make_ptr(
                cutlass.Int64,
                tensor_of_abc_ptrs.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            "cute_ptr_of_tensor_of_sfasfb_ptrs": make_ptr(
                cutlass.Int64,
                tensor_of_sfasfb_ptrs.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            "total_num_clusters": total_num_clusters,
            "persistent_blocks": persistent_blocks,
            "num_groups": len(problem_sizes),
            "last_abc_ptrs": [[0, 0, 0] for _ in range(num_groups_local)],
            "last_sfasfb_ptrs": [[0, 0] for _ in range(num_groups_local)],
        }
        _runtime_meta_cache[runtime_key] = runtime_meta
    else:
        tensor_of_problem_sizes = runtime_meta["tensor_of_problem_sizes"]
        tensor_of_cta_prefix = runtime_meta["tensor_of_cta_prefix"]
        tensor_of_tensormap = runtime_meta["tensor_of_tensormap"]
        tensor_of_abc_ptrs = runtime_meta["tensor_of_abc_ptrs"]
        tensor_of_sfasfb_ptrs = runtime_meta["tensor_of_sfasfb_ptrs"]
        host_abc_ptrs = runtime_meta["host_abc_ptrs"]
        host_sfasfb_ptrs = runtime_meta["host_sfasfb_ptrs"]

    total_num_clusters = runtime_meta["total_num_clusters"]
    persistent_blocks = runtime_meta["persistent_blocks"]
    num_groups = runtime_meta["num_groups"]

    # Avoid rewriting pinned host buffers every invocation; this can race with
    # outstanding async H2D copies in benchmark loops.
    last_abc_ptrs = runtime_meta["last_abc_ptrs"]
    last_sfasfb_ptrs = runtime_meta["last_sfasfb_ptrs"]
    ptrs_changed = False
    for i, ((a, b, c), (sfa_reordered, sfb_reordered), _) in enumerate(
        zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
    ):
        a_ptr = a.data_ptr()
        b_ptr = b.data_ptr()
        c_ptr = c.data_ptr()
        sfa_ptr = sfa_reordered.data_ptr()
        sfb_ptr = sfb_reordered.data_ptr()
        if (
            last_abc_ptrs[i][0] != a_ptr
            or last_abc_ptrs[i][1] != b_ptr
            or last_abc_ptrs[i][2] != c_ptr
            or last_sfasfb_ptrs[i][0] != sfa_ptr
            or last_sfasfb_ptrs[i][1] != sfb_ptr
        ):
            ptrs_changed = True
            last_abc_ptrs[i][0] = a_ptr
            last_abc_ptrs[i][1] = b_ptr
            last_abc_ptrs[i][2] = c_ptr
            last_sfasfb_ptrs[i][0] = sfa_ptr
            last_sfasfb_ptrs[i][1] = sfb_ptr

    if ptrs_changed:
        for i in range(num_groups):
            host_abc_ptrs[i, 0] = last_abc_ptrs[i][0]
            host_abc_ptrs[i, 1] = last_abc_ptrs[i][1]
            host_abc_ptrs[i, 2] = last_abc_ptrs[i][2]
            host_sfasfb_ptrs[i, 0] = last_sfasfb_ptrs[i][0]
            host_sfasfb_ptrs[i, 1] = last_sfasfb_ptrs[i][1]
        tensor_of_abc_ptrs.copy_(host_abc_ptrs, non_blocking=True)
        tensor_of_sfasfb_ptrs.copy_(host_sfasfb_ptrs, non_blocking=True)

    # Create CuTe pointers to the metadata tensors that will be passed to the kernel
    # These allow the GPU kernel to read problem sizes and tensor pointers
    cute_ptr_of_tensor_of_abc_ptrs = runtime_meta["cute_ptr_of_tensor_of_abc_ptrs"]
    cute_ptr_of_tensor_of_sfasfb_ptrs = runtime_meta["cute_ptr_of_tensor_of_sfasfb_ptrs"]
    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32,
        tensor_of_problem_sizes.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_cta_prefix = make_ptr(
        cutlass.Int32,
        tensor_of_cta_prefix.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64,
        tensor_of_tensormap.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    # Launch the JIT-compiled GPU kernel with all prepared data
    # The kernel will perform block-scaled group GEMM: C = A * SFA * B * SFB for all groups
    compiled_func(
        cute_ptr_of_tensor_of_problem_sizes, # Pointer to problem sizes array
        cute_ptr_of_tensor_of_abc_ptrs,      # Pointer to ABC tensor pointers array
        cute_ptr_of_tensor_of_sfasfb_ptrs,   # Pointer to scale factor pointers array
        cute_ptr_of_tensor_of_cta_prefix,    # Pointer to CTA prefix array
        cute_ptr_of_tensor_of_tensormap,     # Pointer to tensormap buffer
        total_num_clusters,                  # Total number of CTAs to launch
        persistent_blocks,                   # Number of persistent CTAs to launch
        problem_sizes,                       # Problem sizes list (for host-side processing)
    )

    res = []
    for i in range(num_groups):
        res.append(output_abc_tensors[i][2])
    return res
