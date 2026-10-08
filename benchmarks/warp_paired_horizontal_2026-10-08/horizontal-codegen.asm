
/home/lilith/work/zen/rav1d-safe/target/lead-warp-paired-current-safe/release/examples/profile_ivf:     file format elf64-x86-64


Disassembly of section .text:

00000000002b2fb0 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc>:
  2b2fb0:	55                   	push   %rbp
  2b2fb1:	41 57                	push   %r15
  2b2fb3:	41 56                	push   %r14
  2b2fb5:	41 55                	push   %r13
  2b2fb7:	41 54                	push   %r12
  2b2fb9:	53                   	push   %rbx
  2b2fba:	48 83 ec 28          	sub    $0x28,%rsp
  2b2fbe:	48 8d 41 fd          	lea    -0x3(%rcx),%rax
  2b2fc2:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  2b2fc7:	41 bd 00 02 00 00    	mov    $0x200,%r13d
  2b2fcd:	44 03 6c 24 68       	add    0x68(%rsp),%r13d
  2b2fd2:	4b 8d 04 40          	lea    (%r8,%r8,2),%rax
  2b2fd6:	48 29 c1             	sub    %rax,%rcx
  2b2fd9:	48 c7 c5 f1 ff ff ff 	mov    $0xfffffffffffffff1,%rbp
  2b2fe0:	45 31 f6             	xor    %r14d,%r14d
  2b2fe3:	4c 8d 3d 86 61 de ff 	lea    -0x219e7a(%rip),%r15        # 99170 <_RNvNtNtCslG8uCnrPz0o_10rav1d_safe3src6tables20dav1d_mc_warp_filter>
  2b2fea:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
  2b2fef:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
  2b2ff4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  2b2ffb:	00 00 00 00 00 
  2b3000:	44 89 e8             	mov    %r13d,%eax
  2b3003:	c1 f8 0a             	sar    $0xa,%eax
  2b3006:	83 c0 40             	add    $0x40,%eax
  2b3009:	4c 63 d8             	movslq %eax,%r11
  2b300c:	41 81 fb c1 00 00 00 	cmp    $0xc1,%r11d
  2b3013:	0f 83 df 03 00 00    	jae    2b33f8 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x448>
  2b3019:	47 8d 24 29          	lea    (%r9,%r13,1),%r12d
  2b301d:	44 89 e0             	mov    %r12d,%eax
  2b3020:	c1 f8 0a             	sar    $0xa,%eax
  2b3023:	83 c0 40             	add    $0x40,%eax
  2b3026:	48 63 d8             	movslq %eax,%rbx
  2b3029:	81 fb c1 00 00 00    	cmp    $0xc1,%ebx
  2b302f:	0f 83 da 03 00 00    	jae    2b340f <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x45f>
  2b3035:	4c 8d 51 fd          	lea    -0x3(%rcx),%r10
  2b3039:	48 8d 41 05          	lea    0x5(%rcx),%rax
  2b303d:	49 83 fa f7          	cmp    $0xfffffffffffffff7,%r10
  2b3041:	0f 87 74 03 00 00    	ja     2b33bb <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x40b>
  2b3047:	48 39 d0             	cmp    %rdx,%rax
  2b304a:	0f 87 6b 03 00 00    	ja     2b33bb <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x40b>
  2b3050:	4c 89 74 24 20       	mov    %r14,0x20(%rsp)
  2b3055:	49 8d 46 fd          	lea    -0x3(%r14),%rax
  2b3059:	49 0f af c0          	imul   %r8,%rax
  2b305d:	48 03 44 24 18       	add    0x18(%rsp),%rax
  2b3062:	48 8d 51 fe          	lea    -0x2(%rcx),%rdx
  2b3066:	48 83 fa f7          	cmp    $0xfffffffffffffff7,%rdx
  2b306a:	0f 87 1b 03 00 00    	ja     2b338b <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3db>
  2b3070:	4c 8d 71 06          	lea    0x6(%rcx),%r14
  2b3074:	4c 3b 74 24 08       	cmp    0x8(%rsp),%r14
  2b3079:	0f 87 0c 03 00 00    	ja     2b338b <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3db>
  2b307f:	c5 fa 7e 44 0e fe    	vmovq  -0x2(%rsi,%rcx,1),%xmm0
  2b3085:	c5 fa 7e 4c 0e fd    	vmovq  -0x3(%rsi,%rcx,1),%xmm1
  2b308b:	c5 f1 6c c0          	vpunpcklqdq %xmm0,%xmm1,%xmm0
  2b308f:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  2b3094:	c4 c1 7a 7e 0c df    	vmovq  (%r15,%rbx,8),%xmm1
  2b309a:	c4 81 7a 7e 14 df    	vmovq  (%r15,%r11,8),%xmm2
  2b30a0:	c5 e9 6c c9          	vpunpcklqdq %xmm1,%xmm2,%xmm1
  2b30a4:	c4 e2 7d 20 c9       	vpmovsxbw %xmm1,%ymm1
  2b30a9:	c5 fd f5 c1          	vpmaddwd %ymm1,%ymm0,%ymm0
  2b30ad:	c4 e2 7d 02 c0       	vphaddd %ymm0,%ymm0,%ymm0
  2b30b2:	c5 fd 70 c8 55       	vpshufd $0x55,%ymm0,%ymm1
  2b30b7:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
  2b30bb:	c4 c1 79 7e c3       	vmovd  %xmm0,%r11d
  2b30c0:	41 83 c3 04          	add    $0x4,%r11d
  2b30c4:	41 c1 eb 03          	shr    $0x3,%r11d
  2b30c8:	66 44 89 5c 6f 1e    	mov    %r11w,0x1e(%rdi,%rbp,2)
  2b30ce:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  2b30d4:	c4 c1 79 7e c3       	vmovd  %xmm0,%r11d
  2b30d9:	41 83 c3 04          	add    $0x4,%r11d
  2b30dd:	41 c1 eb 03          	shr    $0x3,%r11d
  2b30e1:	66 44 89 5c 6f 3c    	mov    %r11w,0x3c(%rdi,%rbp,2)
  2b30e7:	45 01 cc             	add    %r9d,%r12d
  2b30ea:	45 89 e3             	mov    %r12d,%r11d
  2b30ed:	41 c1 fb 0a          	sar    $0xa,%r11d
  2b30f1:	41 83 c3 40          	add    $0x40,%r11d
  2b30f5:	4d 63 db             	movslq %r11d,%r11
  2b30f8:	41 81 fb c1 00 00 00 	cmp    $0xc1,%r11d
  2b30ff:	0f 83 f3 02 00 00    	jae    2b33f8 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x448>
  2b3105:	45 01 cc             	add    %r9d,%r12d
  2b3108:	44 89 e3             	mov    %r12d,%ebx
  2b310b:	c1 fb 0a             	sar    $0xa,%ebx
  2b310e:	83 c3 40             	add    $0x40,%ebx
  2b3111:	48 63 db             	movslq %ebx,%rbx
  2b3114:	81 fb c0 00 00 00    	cmp    $0xc0,%ebx
  2b311a:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  2b311f:	0f 87 ea 02 00 00    	ja     2b340f <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x45f>
  2b3125:	49 83 fa f5          	cmp    $0xfffffffffffffff5,%r10
  2b3129:	0f 87 65 02 00 00    	ja     2b3394 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3e4>
  2b312f:	4c 8d 71 07          	lea    0x7(%rcx),%r14
  2b3133:	49 39 d6             	cmp    %rdx,%r14
  2b3136:	0f 87 58 02 00 00    	ja     2b3394 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3e4>
  2b313c:	4c 8d 71 08          	lea    0x8(%rcx),%r14
  2b3140:	48 83 f9 f7          	cmp    $0xfffffffffffffff7,%rcx
  2b3144:	0f 87 91 02 00 00    	ja     2b33db <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x42b>
  2b314a:	49 39 d6             	cmp    %rdx,%r14
  2b314d:	0f 87 88 02 00 00    	ja     2b33db <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x42b>
  2b3153:	c5 fa 7e 04 0e       	vmovq  (%rsi,%rcx,1),%xmm0
  2b3158:	c5 fa 7e 4c 0e ff    	vmovq  -0x1(%rsi,%rcx,1),%xmm1
  2b315e:	c5 f1 6c c0          	vpunpcklqdq %xmm0,%xmm1,%xmm0
  2b3162:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  2b3167:	c4 c1 7a 7e 0c df    	vmovq  (%r15,%rbx,8),%xmm1
  2b316d:	c4 81 7a 7e 14 df    	vmovq  (%r15,%r11,8),%xmm2
  2b3173:	c5 e9 6c c9          	vpunpcklqdq %xmm1,%xmm2,%xmm1
  2b3177:	c4 e2 7d 20 c9       	vpmovsxbw %xmm1,%ymm1
  2b317c:	c5 fd f5 c1          	vpmaddwd %ymm1,%ymm0,%ymm0
  2b3180:	c4 e2 7d 02 c0       	vphaddd %ymm0,%ymm0,%ymm0
  2b3185:	c5 fd 70 c8 55       	vpshufd $0x55,%ymm0,%ymm1
  2b318a:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
  2b318e:	c4 c1 79 7e c3       	vmovd  %xmm0,%r11d
  2b3193:	41 83 c3 04          	add    $0x4,%r11d
  2b3197:	41 c1 eb 03          	shr    $0x3,%r11d
  2b319b:	66 44 89 5c 6f 5a    	mov    %r11w,0x5a(%rdi,%rbp,2)
  2b31a1:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  2b31a7:	c4 c1 79 7e c3       	vmovd  %xmm0,%r11d
  2b31ac:	41 83 c3 04          	add    $0x4,%r11d
  2b31b0:	41 c1 eb 03          	shr    $0x3,%r11d
  2b31b4:	66 44 89 5c 6f 78    	mov    %r11w,0x78(%rdi,%rbp,2)
  2b31ba:	45 01 cc             	add    %r9d,%r12d
  2b31bd:	45 89 e3             	mov    %r12d,%r11d
  2b31c0:	41 c1 fb 0a          	sar    $0xa,%r11d
  2b31c4:	41 83 c3 40          	add    $0x40,%r11d
  2b31c8:	4d 63 db             	movslq %r11d,%r11
  2b31cb:	41 81 fb c0 00 00 00 	cmp    $0xc0,%r11d
  2b31d2:	0f 87 20 02 00 00    	ja     2b33f8 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x448>
  2b31d8:	45 01 cc             	add    %r9d,%r12d
  2b31db:	44 89 e3             	mov    %r12d,%ebx
  2b31de:	c1 fb 0a             	sar    $0xa,%ebx
  2b31e1:	83 c3 40             	add    $0x40,%ebx
  2b31e4:	48 63 db             	movslq %ebx,%rbx
  2b31e7:	81 fb c0 00 00 00    	cmp    $0xc0,%ebx
  2b31ed:	0f 87 1c 02 00 00    	ja     2b340f <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x45f>
  2b31f3:	49 83 fa f3          	cmp    $0xfffffffffffffff3,%r10
  2b31f7:	0f 87 a0 01 00 00    	ja     2b339d <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3ed>
  2b31fd:	4c 8d 71 09          	lea    0x9(%rcx),%r14
  2b3201:	49 39 d6             	cmp    %rdx,%r14
  2b3204:	0f 87 93 01 00 00    	ja     2b339d <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3ed>
  2b320a:	49 83 fa f2          	cmp    $0xfffffffffffffff2,%r10
  2b320e:	0f 87 92 01 00 00    	ja     2b33a6 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3f6>
  2b3214:	4c 8d 71 0a          	lea    0xa(%rcx),%r14
  2b3218:	49 39 d6             	cmp    %rdx,%r14
  2b321b:	0f 87 85 01 00 00    	ja     2b33a6 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x3f6>
  2b3221:	c5 fa 7e 44 0e 02    	vmovq  0x2(%rsi,%rcx,1),%xmm0
  2b3227:	c5 fa 7e 4c 0e 01    	vmovq  0x1(%rsi,%rcx,1),%xmm1
  2b322d:	c5 f1 6c c0          	vpunpcklqdq %xmm0,%xmm1,%xmm0
  2b3231:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  2b3236:	c4 c1 7a 7e 0c df    	vmovq  (%r15,%rbx,8),%xmm1
  2b323c:	c4 81 7a 7e 14 df    	vmovq  (%r15,%r11,8),%xmm2
  2b3242:	c5 e9 6c c9          	vpunpcklqdq %xmm1,%xmm2,%xmm1
  2b3246:	c4 e2 7d 20 c9       	vpmovsxbw %xmm1,%ymm1
  2b324b:	c5 fd f5 c1          	vpmaddwd %ymm1,%ymm0,%ymm0
  2b324f:	c4 e2 7d 02 c0       	vphaddd %ymm0,%ymm0,%ymm0
  2b3254:	c5 fd 70 c8 55       	vpshufd $0x55,%ymm0,%ymm1
  2b3259:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
  2b325d:	c4 c1 79 7e c3       	vmovd  %xmm0,%r11d
  2b3262:	41 83 c3 04          	add    $0x4,%r11d
  2b3266:	41 c1 eb 03          	shr    $0x3,%r11d
  2b326a:	66 44 89 9c 6f 96 00 	mov    %r11w,0x96(%rdi,%rbp,2)
  2b3271:	00 00 
  2b3273:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  2b3279:	c4 c1 79 7e c3       	vmovd  %xmm0,%r11d
  2b327e:	41 83 c3 04          	add    $0x4,%r11d
  2b3282:	41 c1 eb 03          	shr    $0x3,%r11d
  2b3286:	66 44 89 9c 6f b4 00 	mov    %r11w,0xb4(%rdi,%rbp,2)
  2b328d:	00 00 
  2b328f:	45 01 cc             	add    %r9d,%r12d
  2b3292:	45 89 e3             	mov    %r12d,%r11d
  2b3295:	41 c1 fb 0a          	sar    $0xa,%r11d
  2b3299:	41 83 c3 40          	add    $0x40,%r11d
  2b329d:	4d 63 db             	movslq %r11d,%r11
  2b32a0:	41 81 fb c0 00 00 00 	cmp    $0xc0,%r11d
  2b32a7:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
  2b32ac:	0f 87 46 01 00 00    	ja     2b33f8 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x448>
  2b32b2:	45 01 cc             	add    %r9d,%r12d
  2b32b5:	41 c1 fc 0a          	sar    $0xa,%r12d
  2b32b9:	41 83 c4 40          	add    $0x40,%r12d
  2b32bd:	49 63 dc             	movslq %r12d,%rbx
  2b32c0:	81 fb c0 00 00 00    	cmp    $0xc0,%ebx
  2b32c6:	0f 87 43 01 00 00    	ja     2b340f <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x45f>
  2b32cc:	49 83 fa f1          	cmp    $0xfffffffffffffff1,%r10
  2b32d0:	0f 87 da 00 00 00    	ja     2b33b0 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x400>
  2b32d6:	4c 8d 71 0b          	lea    0xb(%rcx),%r14
  2b32da:	49 39 d6             	cmp    %rdx,%r14
  2b32dd:	0f 87 cd 00 00 00    	ja     2b33b0 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x400>
  2b32e3:	49 83 fa f0          	cmp    $0xfffffffffffffff0,%r10
  2b32e7:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
  2b32ec:	0f 87 de 00 00 00    	ja     2b33d0 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x420>
  2b32f2:	4c 8d 51 0c          	lea    0xc(%rcx),%r10
  2b32f6:	49 39 d2             	cmp    %rdx,%r10
  2b32f9:	0f 87 d1 00 00 00    	ja     2b33d0 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x420>
  2b32ff:	c5 fa 7e 44 0e 04    	vmovq  0x4(%rsi,%rcx,1),%xmm0
  2b3305:	c5 fa 7e 4c 0e 03    	vmovq  0x3(%rsi,%rcx,1),%xmm1
  2b330b:	c5 f1 6c c0          	vpunpcklqdq %xmm0,%xmm1,%xmm0
  2b330f:	c4 e2 7d 30 c0       	vpmovzxbw %xmm0,%ymm0
  2b3314:	c4 c1 7a 7e 0c df    	vmovq  (%r15,%rbx,8),%xmm1
  2b331a:	c4 81 7a 7e 14 df    	vmovq  (%r15,%r11,8),%xmm2
  2b3320:	c5 e9 6c c9          	vpunpcklqdq %xmm1,%xmm2,%xmm1
  2b3324:	c4 e2 7d 20 c9       	vpmovsxbw %xmm1,%ymm1
  2b3329:	c5 fd f5 c1          	vpmaddwd %ymm1,%ymm0,%ymm0
  2b332d:	c4 e2 7d 02 c0       	vphaddd %ymm0,%ymm0,%ymm0
  2b3332:	c5 fd 70 c8 55       	vpshufd $0x55,%ymm0,%ymm1
  2b3337:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
  2b333b:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  2b333f:	83 c0 04             	add    $0x4,%eax
  2b3342:	c1 e8 03             	shr    $0x3,%eax
  2b3345:	66 89 84 6f d2 00 00 	mov    %ax,0xd2(%rdi,%rbp,2)
  2b334c:	00 
  2b334d:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
  2b3353:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  2b3357:	83 c0 04             	add    $0x4,%eax
  2b335a:	c1 e8 03             	shr    $0x3,%eax
  2b335d:	66 89 84 6f f0 00 00 	mov    %ax,0xf0(%rdi,%rbp,2)
  2b3364:	00 
  2b3365:	44 03 6c 24 60       	add    0x60(%rsp),%r13d
  2b336a:	4c 01 c1             	add    %r8,%rcx
  2b336d:	49 ff c6             	inc    %r14
  2b3370:	48 ff c5             	inc    %rbp
  2b3373:	0f 85 87 fc ff ff    	jne    2b3000 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x50>
  2b3379:	48 83 c4 28          	add    $0x28,%rsp
  2b337d:	5b                   	pop    %rbx
  2b337e:	41 5c                	pop    %r12
  2b3380:	41 5d                	pop    %r13
  2b3382:	41 5e                	pop    %r14
  2b3384:	41 5f                	pop    %r15
  2b3386:	5d                   	pop    %rbp
  2b3387:	c5 f8 77             	vzeroupper
  2b338a:	c3                   	ret
  2b338b:	48 83 c0 09          	add    $0x9,%rax
  2b338f:	49 89 c6             	mov    %rax,%r14
  2b3392:	eb 4a                	jmp    2b33de <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x42e>
  2b3394:	48 83 c0 0a          	add    $0xa,%rax
  2b3398:	48 ff c9             	dec    %rcx
  2b339b:	eb 1b                	jmp    2b33b8 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x408>
  2b339d:	48 83 c0 0c          	add    $0xc,%rax
  2b33a1:	48 ff c1             	inc    %rcx
  2b33a4:	eb 12                	jmp    2b33b8 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x408>
  2b33a6:	48 83 c0 0d          	add    $0xd,%rax
  2b33aa:	48 83 c1 02          	add    $0x2,%rcx
  2b33ae:	eb 28                	jmp    2b33d8 <_RNvNtNtNtCslG8uCnrPz0o_10rav1d_safe3src9safe_simd2mc16warp_h_pass_8bpc+0x428>
  2b33b0:	48 83 c0 0e          	add    $0xe,%rax
  2b33b4:	48 83 c1 03          	add    $0x3,%rcx
  2b33b8:	49 89 ca             	mov    %rcx,%r10
  2b33bb:	48 8d 0d d6 a3 1d 00 	lea    0x1da3d6(%rip),%rcx        # 48d798 <__frame_dummy_init_array_entry+0x9bb8>
  2b33c2:	4c 89 d7             	mov    %r10,%rdi
  2b33c5:	48 89 c6             	mov    %rax,%rsi
  2b33c8:	c5 f8 77             	vzeroupper
  2b33cb:	e8 f0 b9 e6 ff       	call   11edc0 <_RNvNtNtCsevLNFiNqfJP_4core5slice5index16slice_index_fail>
  2b33d0:	48 83 c0 0f          	add    $0xf,%rax
  2b33d4:	48 83 c1 04          	add    $0x4,%rcx
  2b33d8:	49 89 c6             	mov    %rax,%r14
  2b33db:	48 89 ca             	mov    %rcx,%rdx
  2b33de:	48 8d 0d cb a3 1d 00 	lea    0x1da3cb(%rip),%rcx        # 48d7b0 <__frame_dummy_init_array_entry+0x9bd0>
  2b33e5:	48 89 d7             	mov    %rdx,%rdi
  2b33e8:	4c 89 f6             	mov    %r14,%rsi
  2b33eb:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  2b33f0:	c5 f8 77             	vzeroupper
  2b33f3:	e8 c8 b9 e6 ff       	call   11edc0 <_RNvNtNtCsevLNFiNqfJP_4core5slice5index16slice_index_fail>
  2b33f8:	48 8d 15 e1 a3 1d 00 	lea    0x1da3e1(%rip),%rdx        # 48d7e0 <__frame_dummy_init_array_entry+0x9c00>
  2b33ff:	be c1 00 00 00       	mov    $0xc1,%esi
  2b3404:	4c 89 df             	mov    %r11,%rdi
  2b3407:	c5 f8 77             	vzeroupper
  2b340a:	e8 b9 ba e6 ff       	call   11eec8 <_RNvNtCsevLNFiNqfJP_4core9panicking18panic_bounds_check>
  2b340f:	48 8d 15 e2 a3 1d 00 	lea    0x1da3e2(%rip),%rdx        # 48d7f8 <__frame_dummy_init_array_entry+0x9c18>
  2b3416:	be c1 00 00 00       	mov    $0xc1,%esi
  2b341b:	48 89 df             	mov    %rbx,%rdi
  2b341e:	c5 f8 77             	vzeroupper
  2b3421:	e8 a2 ba e6 ff       	call   11eec8 <_RNvNtCsevLNFiNqfJP_4core9panicking18panic_bounds_check>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
