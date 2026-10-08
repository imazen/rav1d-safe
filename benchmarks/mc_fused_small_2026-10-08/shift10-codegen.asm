
/home/lilith/work/zen/rav1d-safe/target/lead-fused-current-safe/release/examples/profile_ivf:     file format elf64-x86-64


Disassembly of section .text:

00000000002a9990 <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>>:
  2a9990:	55                   	push   %rbp
  2a9991:	41 57                	push   %r15
  2a9993:	41 56                	push   %r14
  2a9995:	41 55                	push   %r13
  2a9997:	41 54                	push   %r12
  2a9999:	53                   	push   %rbx
  2a999a:	50                   	push   %rax
  2a999b:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  2a99a0:	44 0f b6 78 02       	movzbl 0x2(%rax),%r15d
  2a99a5:	44 0f b6 60 03       	movzbl 0x3(%rax),%r12d
  2a99aa:	0f b6 68 04          	movzbl 0x4(%rax),%ebp
  2a99ae:	44 0f b6 70 05       	movzbl 0x5(%rax),%r14d
  2a99b3:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  2a99b7:	c5 fe 7f 47 20       	vmovdqu %ymm0,0x20(%rdi)
  2a99bc:	c5 fe 7f 07          	vmovdqu %ymm0,(%rdi)
  2a99c0:	49 89 ca             	mov    %rcx,%r10
  2a99c3:	4d 29 c2             	sub    %r8,%r10
  2a99c6:	49 8d 5a ff          	lea    -0x1(%r10),%rbx
  2a99ca:	49 83 c2 06          	add    $0x6,%r10
  2a99ce:	49 39 da             	cmp    %rbx,%r10
  2a99d1:	0f 82 2b 04 00 00    	jb     2a9e02 <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x472>
  2a99d7:	49 39 d2             	cmp    %rdx,%r10
  2a99da:	0f 87 3a 04 00 00    	ja     2a9e1a <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48a>
  2a99e0:	48 8d 41 ff          	lea    -0x1(%rcx),%rax
  2a99e4:	4c 8d 51 06          	lea    0x6(%rcx),%r10
  2a99e8:	49 39 c2             	cmp    %rax,%r10
  2a99eb:	0f 82 2c 04 00 00    	jb     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a99f1:	49 39 d2             	cmp    %rdx,%r10
  2a99f4:	0f 87 23 04 00 00    	ja     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a99fa:	49 8d 04 08          	lea    (%r8,%rcx,1),%rax
  2a99fe:	48 ff c8             	dec    %rax
  2a9a01:	4d 8d 14 08          	lea    (%r8,%rcx,1),%r10
  2a9a05:	49 83 c2 06          	add    $0x6,%r10
  2a9a09:	49 39 c2             	cmp    %rax,%r10
  2a9a0c:	0f 82 0b 04 00 00    	jb     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9a12:	49 39 d2             	cmp    %rdx,%r10
  2a9a15:	0f 87 02 04 00 00    	ja     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9a1b:	4e 8d 1c 41          	lea    (%rcx,%r8,2),%r11
  2a9a1f:	49 ff cb             	dec    %r11
  2a9a22:	4e 8d 14 41          	lea    (%rcx,%r8,2),%r10
  2a9a26:	49 83 c2 06          	add    $0x6,%r10
  2a9a2a:	4d 39 da             	cmp    %r11,%r10
  2a9a2d:	0f 82 b7 03 00 00    	jb     2a9dea <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x45a>
  2a9a33:	49 39 d2             	cmp    %rdx,%r10
  2a9a36:	0f 87 ae 03 00 00    	ja     2a9dea <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x45a>
  2a9a3c:	45 89 fa             	mov    %r15d,%r10d
  2a9a3f:	41 c0 fa 07          	sar    $0x7,%r10b
  2a9a43:	45 89 e5             	mov    %r12d,%r13d
  2a9a46:	41 c0 fd 07          	sar    $0x7,%r13b
  2a9a4a:	45 0f b6 ed          	movzbl %r13b,%r13d
  2a9a4e:	41 c1 e5 18          	shl    $0x18,%r13d
  2a9a52:	41 c1 e4 10          	shl    $0x10,%r12d
  2a9a56:	45 09 ec             	or     %r13d,%r12d
  2a9a59:	45 0f b6 ea          	movzbl %r10b,%r13d
  2a9a5d:	41 c1 e5 08          	shl    $0x8,%r13d
  2a9a61:	45 09 e5             	or     %r12d,%r13d
  2a9a64:	45 09 fd             	or     %r15d,%r13d
  2a9a67:	41 89 ea             	mov    %ebp,%r10d
  2a9a6a:	41 c0 fa 07          	sar    $0x7,%r10b
  2a9a6e:	45 89 f7             	mov    %r14d,%r15d
  2a9a71:	41 c0 ff 07          	sar    $0x7,%r15b
  2a9a75:	45 0f b6 ff          	movzbl %r15b,%r15d
  2a9a79:	41 c1 e7 18          	shl    $0x18,%r15d
  2a9a7d:	41 c1 e6 10          	shl    $0x10,%r14d
  2a9a81:	45 09 fe             	or     %r15d,%r14d
  2a9a84:	45 0f b6 d2          	movzbl %r10b,%r10d
  2a9a88:	41 c1 e2 08          	shl    $0x8,%r10d
  2a9a8c:	45 09 f2             	or     %r14d,%r10d
  2a9a8f:	41 09 ea             	or     %ebp,%r10d
  2a9a92:	c4 c1 79 6e c1       	vmovd  %r9d,%xmm0
  2a9a97:	c4 e2 79 79 d0       	vpbroadcastw %xmm0,%xmm2
  2a9a9c:	c4 e2 79 79 4c 24 40 	vpbroadcastw 0x40(%rsp),%xmm1
  2a9aa3:	c4 c1 79 6e c5       	vmovd  %r13d,%xmm0
  2a9aa8:	c4 e2 79 58 c0       	vpbroadcastd %xmm0,%xmm0
  2a9aad:	44 0f b7 4c 1e 04    	movzwl 0x4(%rsi,%rbx,1),%r9d
  2a9ab3:	44 0f b6 74 1e 06    	movzbl 0x6(%rsi,%rbx,1),%r14d
  2a9ab9:	41 c1 e6 10          	shl    $0x10,%r14d
  2a9abd:	45 09 ce             	or     %r9d,%r14d
  2a9ac0:	49 c1 e6 20          	shl    $0x20,%r14
  2a9ac4:	44 8b 0c 1e          	mov    (%rsi,%rbx,1),%r9d
  2a9ac8:	4d 09 f1             	or     %r14,%r9
  2a9acb:	c4 c1 f9 6e d9       	vmovq  %r9,%xmm3
  2a9ad0:	c4 e2 61 00 25 27 40 	vpshufb -0x27bfd9(%rip),%xmm3,%xmm4        # 2db00 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x5e0>
  2a9ad7:	d8 ff 
  2a9ad9:	c4 e2 59 04 e2       	vpmaddubsw %xmm2,%xmm4,%xmm4
  2a9ade:	c4 e2 61 00 1d c9 3e 	vpshufb -0x27c137(%rip),%xmm3,%xmm3        # 2d9b0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x490>
  2a9ae5:	d8 ff 
  2a9ae7:	c4 e2 61 04 d9       	vpmaddubsw %xmm1,%xmm3,%xmm3
  2a9aec:	c5 d9 fd db          	vpaddw %xmm3,%xmm4,%xmm3
  2a9af0:	c5 e1 fd 25 b8 33 d8 	vpaddw -0x27cc48(%rip),%xmm3,%xmm4        # 2ceb0 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x2a028>
  2a9af7:	ff 
  2a9af8:	44 0f b7 4c 0e 03    	movzwl 0x3(%rsi,%rcx,1),%r9d
  2a9afe:	0f b6 5c 0e 05       	movzbl 0x5(%rsi,%rcx,1),%ebx
  2a9b03:	c1 e3 10             	shl    $0x10,%ebx
  2a9b06:	44 09 cb             	or     %r9d,%ebx
  2a9b09:	48 c1 e3 20          	shl    $0x20,%rbx
  2a9b0d:	44 8b 4c 0e ff       	mov    -0x1(%rsi,%rcx,1),%r9d
  2a9b12:	49 09 d9             	or     %rbx,%r9
  2a9b15:	c4 c1 f9 6e d9       	vmovq  %r9,%xmm3
  2a9b1a:	c4 e2 61 00 2d 2d 32 	vpshufb -0x27cdd3(%rip),%xmm3,%xmm5        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2a9b21:	d8 ff 
  2a9b23:	c4 c1 79 6e f2       	vmovd  %r10d,%xmm6
  2a9b28:	c4 e2 51 04 ea       	vpmaddubsw %xmm2,%xmm5,%xmm5
  2a9b2d:	c4 e2 61 00 1d 4a 3a 	vpshufb -0x27c5b6(%rip),%xmm3,%xmm3        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2a9b34:	d8 ff 
  2a9b36:	c4 e2 61 04 d9       	vpmaddubsw %xmm1,%xmm3,%xmm3
  2a9b3b:	c5 d1 fd db          	vpaddw %xmm3,%xmm5,%xmm3
  2a9b3f:	c5 e1 fd 2d 99 3d d8 	vpaddw -0x27c267(%rip),%xmm3,%xmm5        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2a9b46:	ff 
  2a9b47:	c4 e2 79 58 de       	vpbroadcastd %xmm6,%xmm3
  2a9b4c:	44 0f b7 4c 06 04    	movzwl 0x4(%rsi,%rax,1),%r9d
  2a9b52:	44 0f b6 54 06 06    	movzbl 0x6(%rsi,%rax,1),%r10d
  2a9b58:	41 c1 e2 10          	shl    $0x10,%r10d
  2a9b5c:	45 09 ca             	or     %r9d,%r10d
  2a9b5f:	49 c1 e2 20          	shl    $0x20,%r10
  2a9b63:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2a9b66:	4c 09 d0             	or     %r10,%rax
  2a9b69:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  2a9b6e:	c4 e2 49 00 3d d9 31 	vpshufb -0x27ce27(%rip),%xmm6,%xmm7        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2a9b75:	d8 ff 
  2a9b77:	c4 e2 41 04 fa       	vpmaddubsw %xmm2,%xmm7,%xmm7
  2a9b7c:	c4 e2 49 00 35 fb 39 	vpshufb -0x27c605(%rip),%xmm6,%xmm6        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2a9b83:	d8 ff 
  2a9b85:	c4 e2 49 04 f1       	vpmaddubsw %xmm1,%xmm6,%xmm6
  2a9b8a:	c5 c1 fd f6          	vpaddw %xmm6,%xmm7,%xmm6
  2a9b8e:	c5 c9 fd 3d 4a 3d d8 	vpaddw -0x27c2b6(%rip),%xmm6,%xmm7        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2a9b95:	ff 
  2a9b96:	42 0f b7 44 1e 04    	movzwl 0x4(%rsi,%r11,1),%eax
  2a9b9c:	46 0f b6 4c 1e 06    	movzbl 0x6(%rsi,%r11,1),%r9d
  2a9ba2:	41 c1 e1 10          	shl    $0x10,%r9d
  2a9ba6:	41 09 c1             	or     %eax,%r9d
  2a9ba9:	49 c1 e1 20          	shl    $0x20,%r9
  2a9bad:	42 8b 04 1e          	mov    (%rsi,%r11,1),%eax
  2a9bb1:	4c 09 c8             	or     %r9,%rax
  2a9bb4:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  2a9bb9:	c4 62 49 00 05 8e 31 	vpshufb -0x27ce72(%rip),%xmm6,%xmm8        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2a9bc0:	d8 ff 
  2a9bc2:	c5 b1 71 e7 02       	vpsraw $0x2,%xmm7,%xmm9
  2a9bc7:	c4 62 39 04 c2       	vpmaddubsw %xmm2,%xmm8,%xmm8
  2a9bcc:	c4 e2 49 00 35 ab 39 	vpshufb -0x27c655(%rip),%xmm6,%xmm6        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2a9bd3:	d8 ff 
  2a9bd5:	c4 e2 49 04 f1       	vpmaddubsw %xmm1,%xmm6,%xmm6
  2a9bda:	c5 b9 fd f6          	vpaddw %xmm6,%xmm8,%xmm6
  2a9bde:	c5 c9 fd 35 fa 3c d8 	vpaddw -0x27c306(%rip),%xmm6,%xmm6        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2a9be5:	ff 
  2a9be6:	c5 b9 71 e6 02       	vpsraw $0x2,%xmm6,%xmm8
  2a9beb:	c5 d9 61 e5          	vpunpcklwd %xmm5,%xmm4,%xmm4
  2a9bef:	c5 d9 71 e4 02       	vpsraw $0x2,%xmm4,%xmm4
  2a9bf4:	c4 c1 31 61 f0       	vpunpcklwd %xmm8,%xmm9,%xmm6
  2a9bf9:	c5 59 f5 c8          	vpmaddwd %xmm0,%xmm4,%xmm9
  2a9bfd:	c5 49 f5 d3          	vpmaddwd %xmm3,%xmm6,%xmm10
  2a9c01:	c4 e2 79 58 25 0a 76 	vpbroadcastd -0x2789f6(%rip),%xmm4        # 31214 <rav1d_safe::src::decode::rav1d_decode_frame_init::quant_dist_lookup_table+0x331c>
  2a9c08:	d8 ff 
  2a9c0a:	c5 29 fe d4          	vpaddd %xmm4,%xmm10,%xmm10
  2a9c0e:	c4 41 31 fe ca       	vpaddd %xmm10,%xmm9,%xmm9
  2a9c13:	c4 c1 31 72 e1 0a    	vpsrad $0xa,%xmm9,%xmm9
  2a9c19:	c5 79 7f 0f          	vmovdqa %xmm9,(%rdi)
  2a9c1d:	4f 8d 0c 40          	lea    (%r8,%r8,2),%r9
  2a9c21:	4a 8d 04 09          	lea    (%rcx,%r9,1),%rax
  2a9c25:	48 ff c8             	dec    %rax
  2a9c28:	4e 8d 14 09          	lea    (%rcx,%r9,1),%r10
  2a9c2c:	49 83 c2 06          	add    $0x6,%r10
  2a9c30:	49 39 c2             	cmp    %rax,%r10
  2a9c33:	0f 82 e4 01 00 00    	jb     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9c39:	49 39 d2             	cmp    %rdx,%r10
  2a9c3c:	0f 87 db 01 00 00    	ja     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9c42:	44 0f b7 4c 06 04    	movzwl 0x4(%rsi,%rax,1),%r9d
  2a9c48:	44 0f b6 54 06 06    	movzbl 0x6(%rsi,%rax,1),%r10d
  2a9c4e:	41 c1 e2 10          	shl    $0x10,%r10d
  2a9c52:	45 09 ca             	or     %r9d,%r10d
  2a9c55:	49 c1 e2 20          	shl    $0x20,%r10
  2a9c59:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2a9c5c:	4c 09 d0             	or     %r10,%rax
  2a9c5f:	c4 61 f9 6e c8       	vmovq  %rax,%xmm9
  2a9c64:	c4 62 31 00 15 e3 30 	vpshufb -0x27cf1d(%rip),%xmm9,%xmm10        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2a9c6b:	d8 ff 
  2a9c6d:	c4 62 29 04 d2       	vpmaddubsw %xmm2,%xmm10,%xmm10
  2a9c72:	c4 62 31 00 0d 05 39 	vpshufb -0x27c6fb(%rip),%xmm9,%xmm9        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2a9c79:	d8 ff 
  2a9c7b:	c4 62 31 04 c9       	vpmaddubsw %xmm1,%xmm9,%xmm9
  2a9c80:	c4 41 29 fd c9       	vpaddw %xmm9,%xmm10,%xmm9
  2a9c85:	c5 31 fd 0d 53 3c d8 	vpaddw -0x27c3ad(%rip),%xmm9,%xmm9        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2a9c8c:	ff 
  2a9c8d:	c4 c1 29 71 e1 02    	vpsraw $0x2,%xmm9,%xmm10
  2a9c93:	c5 d1 61 ef          	vpunpcklwd %xmm7,%xmm5,%xmm5
  2a9c97:	c5 c1 71 e5 02       	vpsraw $0x2,%xmm5,%xmm7
  2a9c9c:	c4 c1 39 61 ea       	vpunpcklwd %xmm10,%xmm8,%xmm5
  2a9ca1:	c5 c1 f5 f8          	vpmaddwd %xmm0,%xmm7,%xmm7
  2a9ca5:	c5 51 f5 c3          	vpmaddwd %xmm3,%xmm5,%xmm8
  2a9ca9:	c5 c1 fe fc          	vpaddd %xmm4,%xmm7,%xmm7
  2a9cad:	c5 b9 fe ff          	vpaddd %xmm7,%xmm8,%xmm7
  2a9cb1:	c5 c1 72 e7 0a       	vpsrad $0xa,%xmm7,%xmm7
  2a9cb6:	c5 f9 7f 7f 10       	vmovdqa %xmm7,0x10(%rdi)
  2a9cbb:	4a 8d 04 81          	lea    (%rcx,%r8,4),%rax
  2a9cbf:	48 ff c8             	dec    %rax
  2a9cc2:	4e 8d 14 81          	lea    (%rcx,%r8,4),%r10
  2a9cc6:	49 83 c2 06          	add    $0x6,%r10
  2a9cca:	49 39 c2             	cmp    %rax,%r10
  2a9ccd:	0f 82 4a 01 00 00    	jb     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9cd3:	49 39 d2             	cmp    %rdx,%r10
  2a9cd6:	0f 87 41 01 00 00    	ja     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9cdc:	44 0f b7 4c 06 04    	movzwl 0x4(%rsi,%rax,1),%r9d
  2a9ce2:	44 0f b6 54 06 06    	movzbl 0x6(%rsi,%rax,1),%r10d
  2a9ce8:	41 c1 e2 10          	shl    $0x10,%r10d
  2a9cec:	45 09 ca             	or     %r9d,%r10d
  2a9cef:	49 c1 e2 20          	shl    $0x20,%r10
  2a9cf3:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2a9cf6:	4c 09 d0             	or     %r10,%rax
  2a9cf9:	c4 e1 f9 6e f8       	vmovq  %rax,%xmm7
  2a9cfe:	c4 62 41 00 05 49 30 	vpshufb -0x27cfb7(%rip),%xmm7,%xmm8        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2a9d05:	d8 ff 
  2a9d07:	c4 62 39 04 c2       	vpmaddubsw %xmm2,%xmm8,%xmm8
  2a9d0c:	c4 e2 41 00 3d 6b 38 	vpshufb -0x27c795(%rip),%xmm7,%xmm7        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2a9d13:	d8 ff 
  2a9d15:	c4 e2 41 04 f9       	vpmaddubsw %xmm1,%xmm7,%xmm7
  2a9d1a:	c5 b9 fd ff          	vpaddw %xmm7,%xmm8,%xmm7
  2a9d1e:	c5 c1 fd 3d ba 3b d8 	vpaddw -0x27c446(%rip),%xmm7,%xmm7        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2a9d25:	ff 
  2a9d26:	c5 31 61 c7          	vpunpcklwd %xmm7,%xmm9,%xmm8
  2a9d2a:	c4 c1 39 71 e0 02    	vpsraw $0x2,%xmm8,%xmm8
  2a9d30:	c5 c9 f5 f0          	vpmaddwd %xmm0,%xmm6,%xmm6
  2a9d34:	c5 39 f5 c3          	vpmaddwd %xmm3,%xmm8,%xmm8
  2a9d38:	c5 c9 fe f4          	vpaddd %xmm4,%xmm6,%xmm6
  2a9d3c:	c5 b9 fe f6          	vpaddd %xmm6,%xmm8,%xmm6
  2a9d40:	c5 c9 72 e6 0a       	vpsrad $0xa,%xmm6,%xmm6
  2a9d45:	c5 f9 7f 77 20       	vmovdqa %xmm6,0x20(%rdi)
  2a9d4a:	4f 8d 04 80          	lea    (%r8,%r8,4),%r8
  2a9d4e:	4a 8d 04 01          	lea    (%rcx,%r8,1),%rax
  2a9d52:	48 ff c8             	dec    %rax
  2a9d55:	4e 8d 14 01          	lea    (%rcx,%r8,1),%r10
  2a9d59:	49 83 c2 06          	add    $0x6,%r10
  2a9d5d:	49 39 c2             	cmp    %rax,%r10
  2a9d60:	0f 82 b7 00 00 00    	jb     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9d66:	49 39 d2             	cmp    %rdx,%r10
  2a9d69:	0f 87 ae 00 00 00    	ja     2a9e1d <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<10>+0x48d>
  2a9d6f:	0f b7 4c 06 04       	movzwl 0x4(%rsi,%rax,1),%ecx
  2a9d74:	0f b6 54 06 06       	movzbl 0x6(%rsi,%rax,1),%edx
  2a9d79:	c1 e2 10             	shl    $0x10,%edx
  2a9d7c:	09 ca                	or     %ecx,%edx
  2a9d7e:	48 c1 e2 20          	shl    $0x20,%rdx
  2a9d82:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2a9d85:	48 09 d0             	or     %rdx,%rax
  2a9d88:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  2a9d8d:	c4 62 49 00 05 6a 3d 	vpshufb -0x27c296(%rip),%xmm6,%xmm8        # 2db00 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x5e0>
  2a9d94:	d8 ff 
  2a9d96:	c4 e2 39 04 d2       	vpmaddubsw %xmm2,%xmm8,%xmm2
  2a9d9b:	c4 e2 49 00 35 0c 3c 	vpshufb -0x27c3f4(%rip),%xmm6,%xmm6        # 2d9b0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x490>
  2a9da2:	d8 ff 
  2a9da4:	c4 e2 49 04 c9       	vpmaddubsw %xmm1,%xmm6,%xmm1
  2a9da9:	c5 e9 fd c9          	vpaddw %xmm1,%xmm2,%xmm1
  2a9dad:	c5 f1 fd 0d fb 30 d8 	vpaddw -0x27cf05(%rip),%xmm1,%xmm1        # 2ceb0 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x2a028>
  2a9db4:	ff 
  2a9db5:	c5 c1 61 c9          	vpunpcklwd %xmm1,%xmm7,%xmm1
  2a9db9:	c5 f1 71 e1 02       	vpsraw $0x2,%xmm1,%xmm1
  2a9dbe:	c5 d1 f5 c0          	vpmaddwd %xmm0,%xmm5,%xmm0
  2a9dc2:	c5 f1 f5 cb          	vpmaddwd %xmm3,%xmm1,%xmm1
  2a9dc6:	c5 f9 fe c1          	vpaddd %xmm1,%xmm0,%xmm0
  2a9dca:	c5 f9 fe c4          	vpaddd %xmm4,%xmm0,%xmm0
  2a9dce:	c5 f9 72 e0 0a       	vpsrad $0xa,%xmm0,%xmm0
  2a9dd3:	c5 f9 7f 47 30       	vmovdqa %xmm0,0x30(%rdi)
  2a9dd8:	48 83 c4 08          	add    $0x8,%rsp
  2a9ddc:	5b                   	pop    %rbx
  2a9ddd:	41 5c                	pop    %r12
  2a9ddf:	41 5d                	pop    %r13
  2a9de1:	41 5e                	pop    %r14
  2a9de3:	41 5f                	pop    %r15
  2a9de5:	5d                   	pop    %rbp
  2a9de6:	c5 f8 77             	vzeroupper
  2a9de9:	c3                   	ret
  2a9dea:	4c 89 d8             	mov    %r11,%rax
  2a9ded:	48 8d 0d bc 41 1e 00 	lea    0x1e41bc(%rip),%rcx        # 48dfb0 <__frame_dummy_init_array_entry+0x8e80>
  2a9df4:	48 89 c7             	mov    %rax,%rdi
  2a9df7:	4c 89 d6             	mov    %r10,%rsi
  2a9dfa:	c5 f8 77             	vzeroupper
  2a9dfd:	e8 9e 51 e7 ff       	call   11efa0 <core::slice::index::slice_index_fail>
  2a9e02:	48 89 d8             	mov    %rbx,%rax
  2a9e05:	48 8d 0d a4 41 1e 00 	lea    0x1e41a4(%rip),%rcx        # 48dfb0 <__frame_dummy_init_array_entry+0x8e80>
  2a9e0c:	48 89 c7             	mov    %rax,%rdi
  2a9e0f:	4c 89 d6             	mov    %r10,%rsi
  2a9e12:	c5 f8 77             	vzeroupper
  2a9e15:	e8 86 51 e7 ff       	call   11efa0 <core::slice::index::slice_index_fail>
  2a9e1a:	48 89 d8             	mov    %rbx,%rax
  2a9e1d:	48 8d 0d 8c 41 1e 00 	lea    0x1e418c(%rip),%rcx        # 48dfb0 <__frame_dummy_init_array_entry+0x8e80>
  2a9e24:	48 89 c7             	mov    %rax,%rdi
  2a9e27:	4c 89 d6             	mov    %r10,%rsi
  2a9e2a:	c5 f8 77             	vzeroupper
  2a9e2d:	e8 6e 51 e7 ff       	call   11efa0 <core::slice::index::slice_index_fail>
