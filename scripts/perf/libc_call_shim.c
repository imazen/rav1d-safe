#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdint.h>
#define MINREC 1  // record callers of every call; raise to hide small ones
// bucket by size: <=16, <=64, <=256, <=4K, <=64K, >64K
#define NB 6
static atomic_ullong calls[3][NB], bytes[3][NB];
static int bk(size_t n){ return n<=16?0: n<=64?1: n<=256?2: n<=4096?3: n<=65536?4:5; }
#define TB 8192
static struct { atomic_uintptr_t ra; atomic_ullong calls, bytes; int kind; } tab[TB];
static void rec(int kind, void* ra, size_t n){
  if(n<MINREC) return;
  uintptr_t key=(uintptr_t)ra; size_t h=((key>>2)*0x9E3779B97F4A7C15ULL)>>51; // 13 bits
  for(int i=0;i<TB;i++){ size_t j=(h+i)%TB; uintptr_t cur=atomic_load_explicit(&tab[j].ra,memory_order_relaxed);
    if(cur==key && tab[j].kind==kind){ atomic_fetch_add_explicit(&tab[j].calls,1,memory_order_relaxed); atomic_fetch_add_explicit(&tab[j].bytes,n,memory_order_relaxed); return; }
    if(cur==0){ uintptr_t exp=0; if(atomic_compare_exchange_strong(&tab[j].ra,&exp,key)){ tab[j].kind=kind; atomic_fetch_add_explicit(&tab[j].calls,1,memory_order_relaxed); atomic_fetch_add_explicit(&tab[j].bytes,n,memory_order_relaxed); return; } } }
}
static void* (*real_memcpy)(void*,const void*,size_t);
static void* (*real_memmove)(void*,const void*,size_t);
static void* (*real_memset)(void*,int,size_t);
static void init(void){ if(!real_memcpy){ real_memcpy=dlsym(RTLD_NEXT,"memcpy"); real_memmove=dlsym(RTLD_NEXT,"memmove"); real_memset=dlsym(RTLD_NEXT,"memset"); } }
void* memcpy(void*d,const void*s,size_t n){ init(); int b=bk(n); atomic_fetch_add_explicit(&calls[0][b],1,memory_order_relaxed); atomic_fetch_add_explicit(&bytes[0][b],n,memory_order_relaxed); rec(0,__builtin_return_address(0),n); return real_memcpy(d,s,n);}
void* memmove(void*d,const void*s,size_t n){ init(); int b=bk(n); atomic_fetch_add_explicit(&calls[1][b],1,memory_order_relaxed); atomic_fetch_add_explicit(&bytes[1][b],n,memory_order_relaxed); rec(1,__builtin_return_address(0),n); return real_memmove(d,s,n);}
void* memset(void*d,int c,size_t n){ init(); int b=bk(n); atomic_fetch_add_explicit(&calls[2][b],1,memory_order_relaxed); atomic_fetch_add_explicit(&bytes[2][b],n,memory_order_relaxed); rec(2,__builtin_return_address(0),n); return real_memset(d,c,n);}
__attribute__((destructor)) static void report(void){
  const char* nm[3]={"memcpy","memmove","memset"}; const char* bn[NB]={"<=16","<=64","<=256","<=4K","<=64K",">64K"};
  const char* path=getenv("MEMSHIM_OUT"); FILE* f=path?fopen(path,"w"):stderr; if(!f) f=stderr;
  for(int k=0;k<3;k++){ unsigned long long tc=0,tb=0; for(int b=0;b<NB;b++){tc+=calls[k][b];tb+=bytes[k][b];}
    fprintf(f,"%s total calls=%llu bytes=%llu\n",nm[k],tc,tb);
    for(int b=0;b<NB;b++) if(calls[k][b]) fprintf(f,"  %-6s calls=%llu bytes=%llu\n",bn[b],(unsigned long long)calls[k][b],(unsigned long long)bytes[k][b]); }
  Dl_info di;
  for(int i=0;i<TB;i++){ uintptr_t ra=atomic_load(&tab[i].ra); if(!ra) continue; if(!dladdr((void*)ra,&di)) continue;
    fprintf(f,"CALLER %s %s %llx %llu %llu\n",nm[tab[i].kind],di.dli_fname?di.dli_fname:"?",(unsigned long long)(ra-(uintptr_t)di.dli_fbase),(unsigned long long)tab[i].calls,(unsigned long long)tab[i].bytes); }
  if(f!=stderr) fclose(f);
}
