
/home/lilith/work/zen/rav1d-safe/target/lead-fused-current-safe/release/examples/profile_ivf:     file format elf64-x86-64


Disassembly of section .text:

00000000002c0b50 <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>>:
  2c0b50:	55                   	push   %rbp
  2c0b51:	41 57                	push   %r15
  2c0b53:	41 56                	push   %r14
  2c0b55:	41 55                	push   %r13
  2c0b57:	41 54                	push   %r12
  2c0b59:	53                   	push   %rbx
  2c0b5a:	50                   	push   %rax
  2c0b5b:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  2c0b60:	44 0f b6 78 02       	movzbl 0x2(%rax),%r15d
  2c0b65:	44 0f b6 60 03       	movzbl 0x3(%rax),%r12d
  2c0b6a:	0f b6 68 04          	movzbl 0x4(%rax),%ebp
  2c0b6e:	44 0f b6 70 05       	movzbl 0x5(%rax),%r14d
  2c0b73:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  2c0b77:	c5 fe 7f 47 20       	vmovdqu %ymm0,0x20(%rdi)
  2c0b7c:	c5 fe 7f 07          	vmovdqu %ymm0,(%rdi)
  2c0b80:	49 89 ca             	mov    %rcx,%r10
  2c0b83:	4d 29 c2             	sub    %r8,%r10
  2c0b86:	49 8d 5a ff          	lea    -0x1(%r10),%rbx
  2c0b8a:	49 83 c2 06          	add    $0x6,%r10
  2c0b8e:	49 39 da             	cmp    %rbx,%r10
  2c0b91:	0f 82 2b 04 00 00    	jb     2c0fc2 <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x472>
  2c0b97:	49 39 d2             	cmp    %rdx,%r10
  2c0b9a:	0f 87 3a 04 00 00    	ja     2c0fda <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48a>
  2c0ba0:	48 8d 41 ff          	lea    -0x1(%rcx),%rax
  2c0ba4:	4c 8d 51 06          	lea    0x6(%rcx),%r10
  2c0ba8:	49 39 c2             	cmp    %rax,%r10
  2c0bab:	0f 82 2c 04 00 00    	jb     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0bb1:	49 39 d2             	cmp    %rdx,%r10
  2c0bb4:	0f 87 23 04 00 00    	ja     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0bba:	49 8d 04 08          	lea    (%r8,%rcx,1),%rax
  2c0bbe:	48 ff c8             	dec    %rax
  2c0bc1:	4d 8d 14 08          	lea    (%r8,%rcx,1),%r10
  2c0bc5:	49 83 c2 06          	add    $0x6,%r10
  2c0bc9:	49 39 c2             	cmp    %rax,%r10
  2c0bcc:	0f 82 0b 04 00 00    	jb     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0bd2:	49 39 d2             	cmp    %rdx,%r10
  2c0bd5:	0f 87 02 04 00 00    	ja     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0bdb:	4e 8d 1c 41          	lea    (%rcx,%r8,2),%r11
  2c0bdf:	49 ff cb             	dec    %r11
  2c0be2:	4e 8d 14 41          	lea    (%rcx,%r8,2),%r10
  2c0be6:	49 83 c2 06          	add    $0x6,%r10
  2c0bea:	4d 39 da             	cmp    %r11,%r10
  2c0bed:	0f 82 b7 03 00 00    	jb     2c0faa <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x45a>
  2c0bf3:	49 39 d2             	cmp    %rdx,%r10
  2c0bf6:	0f 87 ae 03 00 00    	ja     2c0faa <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x45a>
  2c0bfc:	45 89 fa             	mov    %r15d,%r10d
  2c0bff:	41 c0 fa 07          	sar    $0x7,%r10b
  2c0c03:	45 89 e5             	mov    %r12d,%r13d
  2c0c06:	41 c0 fd 07          	sar    $0x7,%r13b
  2c0c0a:	45 0f b6 ed          	movzbl %r13b,%r13d
  2c0c0e:	41 c1 e5 18          	shl    $0x18,%r13d
  2c0c12:	41 c1 e4 10          	shl    $0x10,%r12d
  2c0c16:	45 09 ec             	or     %r13d,%r12d
  2c0c19:	45 0f b6 ea          	movzbl %r10b,%r13d
  2c0c1d:	41 c1 e5 08          	shl    $0x8,%r13d
  2c0c21:	45 09 e5             	or     %r12d,%r13d
  2c0c24:	45 09 fd             	or     %r15d,%r13d
  2c0c27:	41 89 ea             	mov    %ebp,%r10d
  2c0c2a:	41 c0 fa 07          	sar    $0x7,%r10b
  2c0c2e:	45 89 f7             	mov    %r14d,%r15d
  2c0c31:	41 c0 ff 07          	sar    $0x7,%r15b
  2c0c35:	45 0f b6 ff          	movzbl %r15b,%r15d
  2c0c39:	41 c1 e7 18          	shl    $0x18,%r15d
  2c0c3d:	41 c1 e6 10          	shl    $0x10,%r14d
  2c0c41:	45 09 fe             	or     %r15d,%r14d
  2c0c44:	45 0f b6 d2          	movzbl %r10b,%r10d
  2c0c48:	41 c1 e2 08          	shl    $0x8,%r10d
  2c0c4c:	45 09 f2             	or     %r14d,%r10d
  2c0c4f:	41 09 ea             	or     %ebp,%r10d
  2c0c52:	c4 c1 79 6e c1       	vmovd  %r9d,%xmm0
  2c0c57:	c4 e2 79 79 d0       	vpbroadcastw %xmm0,%xmm2
  2c0c5c:	c4 e2 79 79 4c 24 40 	vpbroadcastw 0x40(%rsp),%xmm1
  2c0c63:	c4 c1 79 6e c5       	vmovd  %r13d,%xmm0
  2c0c68:	c4 e2 79 58 c0       	vpbroadcastd %xmm0,%xmm0
  2c0c6d:	44 0f b7 4c 1e 04    	movzwl 0x4(%rsi,%rbx,1),%r9d
  2c0c73:	44 0f b6 74 1e 06    	movzbl 0x6(%rsi,%rbx,1),%r14d
  2c0c79:	41 c1 e6 10          	shl    $0x10,%r14d
  2c0c7d:	45 09 ce             	or     %r9d,%r14d
  2c0c80:	49 c1 e6 20          	shl    $0x20,%r14
  2c0c84:	44 8b 0c 1e          	mov    (%rsi,%rbx,1),%r9d
  2c0c88:	4d 09 f1             	or     %r14,%r9
  2c0c8b:	c4 c1 f9 6e d9       	vmovq  %r9,%xmm3
  2c0c90:	c4 e2 61 00 25 67 ce 	vpshufb -0x293199(%rip),%xmm3,%xmm4        # 2db00 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x5e0>
  2c0c97:	d6 ff 
  2c0c99:	c4 e2 59 04 e2       	vpmaddubsw %xmm2,%xmm4,%xmm4
  2c0c9e:	c4 e2 61 00 1d 09 cd 	vpshufb -0x2932f7(%rip),%xmm3,%xmm3        # 2d9b0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x490>
  2c0ca5:	d6 ff 
  2c0ca7:	c4 e2 61 04 d9       	vpmaddubsw %xmm1,%xmm3,%xmm3
  2c0cac:	c5 d9 fd db          	vpaddw %xmm3,%xmm4,%xmm3
  2c0cb0:	c5 e1 fd 25 f8 c1 d6 	vpaddw -0x293e08(%rip),%xmm3,%xmm4        # 2ceb0 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x2a028>
  2c0cb7:	ff 
  2c0cb8:	44 0f b7 4c 0e 03    	movzwl 0x3(%rsi,%rcx,1),%r9d
  2c0cbe:	0f b6 5c 0e 05       	movzbl 0x5(%rsi,%rcx,1),%ebx
  2c0cc3:	c1 e3 10             	shl    $0x10,%ebx
  2c0cc6:	44 09 cb             	or     %r9d,%ebx
  2c0cc9:	48 c1 e3 20          	shl    $0x20,%rbx
  2c0ccd:	44 8b 4c 0e ff       	mov    -0x1(%rsi,%rcx,1),%r9d
  2c0cd2:	49 09 d9             	or     %rbx,%r9
  2c0cd5:	c4 c1 f9 6e d9       	vmovq  %r9,%xmm3
  2c0cda:	c4 e2 61 00 2d 6d c0 	vpshufb -0x293f93(%rip),%xmm3,%xmm5        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2c0ce1:	d6 ff 
  2c0ce3:	c4 c1 79 6e f2       	vmovd  %r10d,%xmm6
  2c0ce8:	c4 e2 51 04 ea       	vpmaddubsw %xmm2,%xmm5,%xmm5
  2c0ced:	c4 e2 61 00 1d 8a c8 	vpshufb -0x293776(%rip),%xmm3,%xmm3        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2c0cf4:	d6 ff 
  2c0cf6:	c4 e2 61 04 d9       	vpmaddubsw %xmm1,%xmm3,%xmm3
  2c0cfb:	c5 d1 fd db          	vpaddw %xmm3,%xmm5,%xmm3
  2c0cff:	c5 e1 fd 2d d9 cb d6 	vpaddw -0x293427(%rip),%xmm3,%xmm5        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2c0d06:	ff 
  2c0d07:	c4 e2 79 58 de       	vpbroadcastd %xmm6,%xmm3
  2c0d0c:	44 0f b7 4c 06 04    	movzwl 0x4(%rsi,%rax,1),%r9d
  2c0d12:	44 0f b6 54 06 06    	movzbl 0x6(%rsi,%rax,1),%r10d
  2c0d18:	41 c1 e2 10          	shl    $0x10,%r10d
  2c0d1c:	45 09 ca             	or     %r9d,%r10d
  2c0d1f:	49 c1 e2 20          	shl    $0x20,%r10
  2c0d23:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2c0d26:	4c 09 d0             	or     %r10,%rax
  2c0d29:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  2c0d2e:	c4 e2 49 00 3d 19 c0 	vpshufb -0x293fe7(%rip),%xmm6,%xmm7        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2c0d35:	d6 ff 
  2c0d37:	c4 e2 41 04 fa       	vpmaddubsw %xmm2,%xmm7,%xmm7
  2c0d3c:	c4 e2 49 00 35 3b c8 	vpshufb -0x2937c5(%rip),%xmm6,%xmm6        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2c0d43:	d6 ff 
  2c0d45:	c4 e2 49 04 f1       	vpmaddubsw %xmm1,%xmm6,%xmm6
  2c0d4a:	c5 c1 fd f6          	vpaddw %xmm6,%xmm7,%xmm6
  2c0d4e:	c5 c9 fd 3d 8a cb d6 	vpaddw -0x293476(%rip),%xmm6,%xmm7        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2c0d55:	ff 
  2c0d56:	42 0f b7 44 1e 04    	movzwl 0x4(%rsi,%r11,1),%eax
  2c0d5c:	46 0f b6 4c 1e 06    	movzbl 0x6(%rsi,%r11,1),%r9d
  2c0d62:	41 c1 e1 10          	shl    $0x10,%r9d
  2c0d66:	41 09 c1             	or     %eax,%r9d
  2c0d69:	49 c1 e1 20          	shl    $0x20,%r9
  2c0d6d:	42 8b 04 1e          	mov    (%rsi,%r11,1),%eax
  2c0d71:	4c 09 c8             	or     %r9,%rax
  2c0d74:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  2c0d79:	c4 62 49 00 05 ce bf 	vpshufb -0x294032(%rip),%xmm6,%xmm8        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2c0d80:	d6 ff 
  2c0d82:	c5 b1 71 e7 02       	vpsraw $0x2,%xmm7,%xmm9
  2c0d87:	c4 62 39 04 c2       	vpmaddubsw %xmm2,%xmm8,%xmm8
  2c0d8c:	c4 e2 49 00 35 eb c7 	vpshufb -0x293815(%rip),%xmm6,%xmm6        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2c0d93:	d6 ff 
  2c0d95:	c4 e2 49 04 f1       	vpmaddubsw %xmm1,%xmm6,%xmm6
  2c0d9a:	c5 b9 fd f6          	vpaddw %xmm6,%xmm8,%xmm6
  2c0d9e:	c5 c9 fd 35 3a cb d6 	vpaddw -0x2934c6(%rip),%xmm6,%xmm6        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2c0da5:	ff 
  2c0da6:	c5 b9 71 e6 02       	vpsraw $0x2,%xmm6,%xmm8
  2c0dab:	c5 d9 61 e5          	vpunpcklwd %xmm5,%xmm4,%xmm4
  2c0daf:	c5 d9 71 e4 02       	vpsraw $0x2,%xmm4,%xmm4
  2c0db4:	c4 c1 31 61 f0       	vpunpcklwd %xmm8,%xmm9,%xmm6
  2c0db9:	c5 59 f5 c8          	vpmaddwd %xmm0,%xmm4,%xmm9
  2c0dbd:	c5 49 f5 d3          	vpmaddwd %xmm3,%xmm6,%xmm10
  2c0dc1:	c4 e2 79 58 25 4a 06 	vpbroadcastd -0x28f9b6(%rip),%xmm4        # 31414 <rav1d_safe::src::decode::rav1d_decode_frame_init::quant_dist_lookup_table+0x351c>
  2c0dc8:	d7 ff 
  2c0dca:	c5 29 fe d4          	vpaddd %xmm4,%xmm10,%xmm10
  2c0dce:	c4 41 31 fe ca       	vpaddd %xmm10,%xmm9,%xmm9
  2c0dd3:	c4 c1 31 72 e1 06    	vpsrad $0x6,%xmm9,%xmm9
  2c0dd9:	c5 79 7f 0f          	vmovdqa %xmm9,(%rdi)
  2c0ddd:	4f 8d 0c 40          	lea    (%r8,%r8,2),%r9
  2c0de1:	4a 8d 04 09          	lea    (%rcx,%r9,1),%rax
  2c0de5:	48 ff c8             	dec    %rax
  2c0de8:	4e 8d 14 09          	lea    (%rcx,%r9,1),%r10
  2c0dec:	49 83 c2 06          	add    $0x6,%r10
  2c0df0:	49 39 c2             	cmp    %rax,%r10
  2c0df3:	0f 82 e4 01 00 00    	jb     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0df9:	49 39 d2             	cmp    %rdx,%r10
  2c0dfc:	0f 87 db 01 00 00    	ja     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0e02:	44 0f b7 4c 06 04    	movzwl 0x4(%rsi,%rax,1),%r9d
  2c0e08:	44 0f b6 54 06 06    	movzbl 0x6(%rsi,%rax,1),%r10d
  2c0e0e:	41 c1 e2 10          	shl    $0x10,%r10d
  2c0e12:	45 09 ca             	or     %r9d,%r10d
  2c0e15:	49 c1 e2 20          	shl    $0x20,%r10
  2c0e19:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2c0e1c:	4c 09 d0             	or     %r10,%rax
  2c0e1f:	c4 61 f9 6e c8       	vmovq  %rax,%xmm9
  2c0e24:	c4 62 31 00 15 23 bf 	vpshufb -0x2940dd(%rip),%xmm9,%xmm10        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2c0e2b:	d6 ff 
  2c0e2d:	c4 62 29 04 d2       	vpmaddubsw %xmm2,%xmm10,%xmm10
  2c0e32:	c4 62 31 00 0d 45 c7 	vpshufb -0x2938bb(%rip),%xmm9,%xmm9        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2c0e39:	d6 ff 
  2c0e3b:	c4 62 31 04 c9       	vpmaddubsw %xmm1,%xmm9,%xmm9
  2c0e40:	c4 41 29 fd c9       	vpaddw %xmm9,%xmm10,%xmm9
  2c0e45:	c5 31 fd 0d 93 ca d6 	vpaddw -0x29356d(%rip),%xmm9,%xmm9        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2c0e4c:	ff 
  2c0e4d:	c4 c1 29 71 e1 02    	vpsraw $0x2,%xmm9,%xmm10
  2c0e53:	c5 d1 61 ef          	vpunpcklwd %xmm7,%xmm5,%xmm5
  2c0e57:	c5 c1 71 e5 02       	vpsraw $0x2,%xmm5,%xmm7
  2c0e5c:	c4 c1 39 61 ea       	vpunpcklwd %xmm10,%xmm8,%xmm5
  2c0e61:	c5 c1 f5 f8          	vpmaddwd %xmm0,%xmm7,%xmm7
  2c0e65:	c5 51 f5 c3          	vpmaddwd %xmm3,%xmm5,%xmm8
  2c0e69:	c5 c1 fe fc          	vpaddd %xmm4,%xmm7,%xmm7
  2c0e6d:	c5 b9 fe ff          	vpaddd %xmm7,%xmm8,%xmm7
  2c0e71:	c5 c1 72 e7 06       	vpsrad $0x6,%xmm7,%xmm7
  2c0e76:	c5 f9 7f 7f 10       	vmovdqa %xmm7,0x10(%rdi)
  2c0e7b:	4a 8d 04 81          	lea    (%rcx,%r8,4),%rax
  2c0e7f:	48 ff c8             	dec    %rax
  2c0e82:	4e 8d 14 81          	lea    (%rcx,%r8,4),%r10
  2c0e86:	49 83 c2 06          	add    $0x6,%r10
  2c0e8a:	49 39 c2             	cmp    %rax,%r10
  2c0e8d:	0f 82 4a 01 00 00    	jb     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0e93:	49 39 d2             	cmp    %rdx,%r10
  2c0e96:	0f 87 41 01 00 00    	ja     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0e9c:	44 0f b7 4c 06 04    	movzwl 0x4(%rsi,%rax,1),%r9d
  2c0ea2:	44 0f b6 54 06 06    	movzbl 0x6(%rsi,%rax,1),%r10d
  2c0ea8:	41 c1 e2 10          	shl    $0x10,%r10d
  2c0eac:	45 09 ca             	or     %r9d,%r10d
  2c0eaf:	49 c1 e2 20          	shl    $0x20,%r10
  2c0eb3:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2c0eb6:	4c 09 d0             	or     %r10,%rax
  2c0eb9:	c4 e1 f9 6e f8       	vmovq  %rax,%xmm7
  2c0ebe:	c4 62 41 00 05 89 be 	vpshufb -0x294177(%rip),%xmm7,%xmm8        # 2cd50 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x29ec8>
  2c0ec5:	d6 ff 
  2c0ec7:	c4 62 39 04 c2       	vpmaddubsw %xmm2,%xmm8,%xmm8
  2c0ecc:	c4 e2 41 00 3d ab c6 	vpshufb -0x293955(%rip),%xmm7,%xmm7        # 2d580 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x60>
  2c0ed3:	d6 ff 
  2c0ed5:	c4 e2 41 04 f9       	vpmaddubsw %xmm1,%xmm7,%xmm7
  2c0eda:	c5 b9 fd ff          	vpaddw %xmm7,%xmm8,%xmm7
  2c0ede:	c5 c1 fd 3d fa c9 d6 	vpaddw -0x293606(%rip),%xmm7,%xmm7        # 2d8e0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x3c0>
  2c0ee5:	ff 
  2c0ee6:	c5 31 61 c7          	vpunpcklwd %xmm7,%xmm9,%xmm8
  2c0eea:	c4 c1 39 71 e0 02    	vpsraw $0x2,%xmm8,%xmm8
  2c0ef0:	c5 c9 f5 f0          	vpmaddwd %xmm0,%xmm6,%xmm6
  2c0ef4:	c5 39 f5 c3          	vpmaddwd %xmm3,%xmm8,%xmm8
  2c0ef8:	c5 c9 fe f4          	vpaddd %xmm4,%xmm6,%xmm6
  2c0efc:	c5 b9 fe f6          	vpaddd %xmm6,%xmm8,%xmm6
  2c0f00:	c5 c9 72 e6 06       	vpsrad $0x6,%xmm6,%xmm6
  2c0f05:	c5 f9 7f 77 20       	vmovdqa %xmm6,0x20(%rdi)
  2c0f0a:	4f 8d 04 80          	lea    (%r8,%r8,4),%r8
  2c0f0e:	4a 8d 04 01          	lea    (%rcx,%r8,1),%rax
  2c0f12:	48 ff c8             	dec    %rax
  2c0f15:	4e 8d 14 01          	lea    (%rcx,%r8,1),%r10
  2c0f19:	49 83 c2 06          	add    $0x6,%r10
  2c0f1d:	49 39 c2             	cmp    %rax,%r10
  2c0f20:	0f 82 b7 00 00 00    	jb     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0f26:	49 39 d2             	cmp    %rdx,%r10
  2c0f29:	0f 87 ae 00 00 00    	ja     2c0fdd <rav1d_safe::src::safe_simd::mc::fused_hv_4x4_8bpc::<6>+0x48d>
  2c0f2f:	0f b7 4c 06 04       	movzwl 0x4(%rsi,%rax,1),%ecx
  2c0f34:	0f b6 54 06 06       	movzbl 0x6(%rsi,%rax,1),%edx
  2c0f39:	c1 e2 10             	shl    $0x10,%edx
  2c0f3c:	09 ca                	or     %ecx,%edx
  2c0f3e:	48 c1 e2 20          	shl    $0x20,%rdx
  2c0f42:	8b 04 06             	mov    (%rsi,%rax,1),%eax
  2c0f45:	48 09 d0             	or     %rdx,%rax
  2c0f48:	c4 e1 f9 6e f0       	vmovq  %rax,%xmm6
  2c0f4d:	c4 62 49 00 05 aa cb 	vpshufb -0x293456(%rip),%xmm6,%xmm8        # 2db00 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x5e0>
  2c0f54:	d6 ff 
  2c0f56:	c4 e2 39 04 d2       	vpmaddubsw %xmm2,%xmm8,%xmm2
  2c0f5b:	c4 e2 49 00 35 4c ca 	vpshufb -0x2935b4(%rip),%xmm6,%xmm6        # 2d9b0 <rav1d_safe::src::safe_simd::filmgrain::__arcane_fgy_inner_16bpc::W+0x490>
  2c0f62:	d6 ff 
  2c0f64:	c4 e2 49 04 c9       	vpmaddubsw %xmm1,%xmm6,%xmm1
  2c0f69:	c5 e9 fd c9          	vpaddw %xmm1,%xmm2,%xmm1
  2c0f6d:	c5 f1 fd 0d 3b bf d6 	vpaddw -0x2940c5(%rip),%xmm1,%xmm1        # 2ceb0 <fastrand::global_rng::RNG::{K#0}::{closure#1}::__RUST_STD_INTERNAL_VAL+0x2a028>
  2c0f74:	ff 
  2c0f75:	c5 c1 61 c9          	vpunpcklwd %xmm1,%xmm7,%xmm1
  2c0f79:	c5 f1 71 e1 02       	vpsraw $0x2,%xmm1,%xmm1
  2c0f7e:	c5 d1 f5 c0          	vpmaddwd %xmm0,%xmm5,%xmm0
  2c0f82:	c5 f1 f5 cb          	vpmaddwd %xmm3,%xmm1,%xmm1
  2c0f86:	c5 f9 fe c1          	vpaddd %xmm1,%xmm0,%xmm0
  2c0f8a:	c5 f9 fe c4          	vpaddd %xmm4,%xmm0,%xmm0
  2c0f8e:	c5 f9 72 e0 06       	vpsrad $0x6,%xmm0,%xmm0
  2c0f93:	c5 f9 7f 47 30       	vmovdqa %xmm0,0x30(%rdi)
  2c0f98:	48 83 c4 08          	add    $0x8,%rsp
  2c0f9c:	5b                   	pop    %rbx
  2c0f9d:	41 5c                	pop    %r12
  2c0f9f:	41 5d                	pop    %r13
  2c0fa1:	41 5e                	pop    %r14
  2c0fa3:	41 5f                	pop    %r15
  2c0fa5:	5d                   	pop    %rbp
  2c0fa6:	c5 f8 77             	vzeroupper
  2c0fa9:	c3                   	ret
  2c0faa:	4c 89 d8             	mov    %r11,%rax
  2c0fad:	48 8d 0d fc cf 1c 00 	lea    0x1ccffc(%rip),%rcx        # 48dfb0 <__frame_dummy_init_array_entry+0x8e80>
  2c0fb4:	48 89 c7             	mov    %rax,%rdi
  2c0fb7:	4c 89 d6             	mov    %r10,%rsi
  2c0fba:	c5 f8 77             	vzeroupper
  2c0fbd:	e8 de df e5 ff       	call   11efa0 <core::slice::index::slice_index_fail>
  2c0fc2:	48 89 d8             	mov    %rbx,%rax
  2c0fc5:	48 8d 0d e4 cf 1c 00 	lea    0x1ccfe4(%rip),%rcx        # 48dfb0 <__frame_dummy_init_array_entry+0x8e80>
  2c0fcc:	48 89 c7             	mov    %rax,%rdi
  2c0fcf:	4c 89 d6             	mov    %r10,%rsi
  2c0fd2:	c5 f8 77             	vzeroupper
  2c0fd5:	e8 c6 df e5 ff       	call   11efa0 <core::slice::index::slice_index_fail>
  2c0fda:	48 89 d8             	mov    %rbx,%rax
  2c0fdd:	48 8d 0d cc cf 1c 00 	lea    0x1ccfcc(%rip),%rcx        # 48dfb0 <__frame_dummy_init_array_entry+0x8e80>
  2c0fe4:	48 89 c7             	mov    %rax,%rdi
  2c0fe7:	4c 89 d6             	mov    %r10,%rsi
  2c0fea:	c5 f8 77             	vzeroupper
  2c0fed:	e8 ae df e5 ff       	call   11efa0 <core::slice::index::slice_index_fail>
