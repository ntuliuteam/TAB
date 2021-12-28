/************************************************* 
* The following kernels are written by Shaun Zhu @ 2021
* yiweifengyan@foxmail.com
* The size is always K*2x64-bit, so it is much simplier than the libpopcnt.h
* The functions use the same number of popcnt instructions inside the main loop for fair comparison
* Reference: https://github.com/kimwalisch/libpopcnt
*/ 

#ifndef LIBPOPCNT_H_ARM
#define LIBPOPCNT_H_ARM
#include <arm_neon.h>
#define popcnt64(uint64_t x) __builtin_popcountll(x);

/*************** Added by Shaun Zhu for reference
vpaddlq_u8 (uint8x16_t __a)
{
  return (uint16x8_t)__builtin_neon_vpaddluv16qi ((int8x16_t) __a);
}
vpaddlq_u16 (uint16x8_t __a)
{
  return (uint32x4_t)__builtin_neon_vpaddluv8hi ((int16x8_t) __a);
}
vpadalq_u32 (uint64x2_t __a, uint32x4_t __b)
{
  return (uint64x2_t)__builtin_neon_vpadaluv4si ((int64x2_t) __a, (int32x4_t) __b);
}
vpadal_u32 (uint64x1_t __a, uint32x2_t __b)
{
  return (uint64x1_t)__builtin_neon_vpadaluv2si ((int64x1_t) __a, (int32x2_t) __b);
}
*/

static inline uint64_t TNNpopKernel(const void* a, const void* b, uint64_t SIZE)
{
  uint64_t i = 0;
  uint64_t size = SIZE;
  uint64_t cnt = 0;
  uint64_t chunk_size = 8;
  //uint64_t chunk_size = 64;
  //const uint8_t* ptr = (const uint8_t*) data;

  uint64_t load_size=4;
  const uint64_t* ptra = (const uint64_t*) a;
  const uint64_t* ptrb = (const uint64_t*) b;

  if (size >= chunk_size)
  {
    uint64_t iters = size / chunk_size;
    uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    uint64x2_t neg = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    uint8x16_t zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));    
    do
    {
      uint8x16_t t0 = zero;
      uint8x16_t t1 = zero;
      uint8x16_t t2 = zero;
      uint8x16_t t3 = zero;
      uint8x16_t r0 = zero;
      uint8x16_t r1 = zero;
      uint8x16_t r2 = zero;
      uint8x16_t r3 = zero;
      /*
       * After every 31 iterations we need to add the
       * temporary sums (t0, t1, t2, t3) to the total sum.
       * We must ensure that the temporary sums <= 255
       * and 31 * 8 bits = 248 which is OK.
       */
      uint64_t limit = (i + 31 < iters) ? i + 31 : iters;

      /* Each iteration processes 64 bytes */
      for (; i < limit; i++)
      {
        //uint8x16x4_t input = vld4q_u8(ptr);
        //ptr += chunk_size;

        uint64x1x4_t a1 = vld4_u64(ptra);
        ptra+=load_size;
        uint64x1x4_t a2 = vld4_u64(ptra);
        ptra+=load_size;
        uint64x1x4_t a3 = vld4_u64(ptra);
        ptra+=load_size;
        uint64x1x4_t a4 = vld4_u64(ptra);
        ptra+=load_size;

        uint64x1x4_t b1 = vld4_u64(ptrb);
        ptrb+=load_size;
        uint64x1x4_t b2 = vld4_u64(ptrb);
        ptrb+=load_size;
        uint64x1x4_t b3 = vld4_u64(ptrb);
        ptrb+=load_size;
        uint64x1x4_t b4 = vld4_u64(ptrb);
        ptrb+=load_size;

        uint8x16_t p1 = vcombine_u64(a1.val[0] ^ b1.val[0], a1.val[2] ^ b1.val[2]);
        uint8x16_t p2 = vcombine_u64(a1.val[1] & b1.val[1], a1.val[3] & b1.val[3]);
        uint8x16_t p3 = vcombine_u64(a2.val[0] ^ b2.val[0], a2.val[2] ^ b2.val[2]);
        uint8x16_t p4 = vcombine_u64(a2.val[1] & b2.val[1], a2.val[3] & b2.val[3]);
        uint8x16_t p5 = vcombine_u64(a3.val[0] ^ b3.val[0], a3.val[2] ^ b3.val[2]);
        uint8x16_t p6 = vcombine_u64(a3.val[1] & b3.val[1], a3.val[3] & b3.val[3]);
        uint8x16_t p7 = vcombine_u64(a4.val[0] ^ b4.val[0], a4.val[2] ^ b4.val[2]);
        uint8x16_t p8 = vcombine_u64(a4.val[1] & b4.val[1], a4.val[3] & b4.val[3]);

        //t0 = vaddq_u8(t0, vcntq_u8(input.val[0]));
        //t1 = vaddq_u8(t1, vcntq_u8(input.val[1]));
        //t2 = vaddq_u8(t2, vcntq_u8(input.val[2]));
        //t3 = vaddq_u8(t3, vcntq_u8(input.val[3]));

        t0 = vaddq_u8(t0, vcntq_u8(p2));
        t1 = vaddq_u8(t1, vcntq_u8(p4));
        t2 = vaddq_u8(t2, vcntq_u8(p6));
        t3 = vaddq_u8(t3, vcntq_u8(p8));
        r0 = vaddq_u8(r0, vcntq_u8(p1 & p2));
        r1 = vaddq_u8(r1, vcntq_u8(p3 & p4));
        r2 = vaddq_u8(r2, vcntq_u8(p5 & p6));
        r3 = vaddq_u8(r3, vcntq_u8(p7 & p8));
      }
      sum = vpadalq(sum, t0);
      sum = vpadalq(sum, t1);
      sum = vpadalq(sum, t2);
      sum = vpadalq(sum, t3);

      neg = vpadalq(neg, r0);
      neg = vpadalq(neg, r1);
      neg = vpadalq(neg, r2);
      neg = vpadalq(neg, r3);
    }
    while (i < iters);

    i = 0;
    size %= chunk_size;

    uint64_t tmp[2];
    vst1q_u64(tmp, sum);
    cnt += tmp[0];
    cnt += tmp[1];
    vst1q_u64(tmp, neg);
    cnt = cnt - tmp[0] - tmp[0];
    cnt = cnt - tmp[1] - tmp[1];
  }

  /* Now there can only be 0, 1, 2, 3 elements left. */
  // deal with two elements
  uint8x16_t t0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
  uint8x16_t r0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
  uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
  uint64x2_t neg = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
  while(size>1){
    uint64x1x4_t a1 = vld4_u64(ptra);
    ptra+=load_size;
    uint64x1x4_t b1 = vld4_u64(ptrb);
    ptrb+=load_size;
    uint8x16_t p1 = vcombine_u64(a1.val[0] ^ b1.val[0], a1.val[2] ^ b1.val[2]);
    uint8x16_t p2 = vcombine_u64(a1.val[1] & b1.val[1], a1.val[3] & b1.val[3]);
    t0 = vaddq_u8(t0, vcntq_u8(p2));
    r0 = vaddq_u8(r0, vcntq_u8(p1 & p2));
    size=size-2;
  }
  sum = vpadalq(sum, t0);
  neg = vpadalq(neg, r0);
  uint64_t tmp[2];
  vst1q_u64(tmp, sum);
  cnt += tmp[0];
  cnt += tmp[1];
  vst1q_u64(tmp, neg);
  cnt = cnt - tmp[0] - tmp[0];
  cnt = cnt - tmp[1] - tmp[1];

  // deal with one element
  if (size>0)
  {
    uint64_t p1=ptra[0] ^ ptrb[0];
    uint64_t p2=ptra[1] & ptrb[1];
    uint64_t nega=popcnt64(p1 & p2);
    cnt += popcnt64(p2) - nega - nega;
  }

  return cnt;
}


static inline uint64_t TBNpopKernel(const void* a, const void* b, uint64_t SIZE)
{
    uint64_t i = 0;
    uint64_t size = SIZE;
    uint64_t cnt = 0;
    uint64_t chunk_size = 8;
    //uint64_t chunk_size = 64;
    //const uint8_t* ptr = (const uint8_t*) data;

    uint64_t load_size = 4;
    const uint64_t* ptra = (const uint64_t*)a;
    const uint64_t* ptrb = (const uint64_t*)b;

    if (size >= chunk_size)
    {
        uint64_t iters = size / chunk_size;
        uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint64x2_t neg = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint8x16_t zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        do
        {
            uint8x16_t t0 = zero;
            uint8x16_t t1 = zero;
            uint8x16_t t2 = zero;
            uint8x16_t t3 = zero;
            uint8x16_t r0 = zero;
            uint8x16_t r1 = zero;
            uint8x16_t r2 = zero;
            uint8x16_t r3 = zero;
            /*
             * After every 31 iterations we need to add the
             * temporary sums (t0, t1, t2, t3) to the total sum.
             * We must ensure that the temporary sums <= 255
             * and 31 * 8 bits = 248 which is OK.
             */
            uint64_t limit = (i + 31 < iters) ? i + 31 : iters;

            /* Each iteration processes 64 bytes */
            for (; i < limit; i++)
            {
                //uint8x16x4_t input = vld4q_u8(ptr);
                //ptr += chunk_size;

                uint64x1x4_t a1 = vld4_u64(ptra);
                ptra += load_size;
                uint64x1x4_t a2 = vld4_u64(ptra);
                ptra += load_size;
                uint64x1x4_t a3 = vld4_u64(ptra);
                ptra += load_size;
                uint64x1x4_t a4 = vld4_u64(ptra);
                ptra += load_size;

                uint64x1x4_t b1 = vld4_u64(ptrb);
                ptrb += load_size;
                uint64x1x4_t b2 = vld4_u64(ptrb);
                ptrb += load_size;

                uint8x16_t p1 = vcombine_u64(a1.val[0] ^ b1.val[0], a1.val[2] ^ b1.val[1]);
                uint8x16_t p2 = vcombine_u64(a1.val[1],  a1.val[3]);
                uint8x16_t p3 = vcombine_u64(a2.val[0] ^ b1.val[2], a2.val[2] ^ b1.val[3]);
                uint8x16_t p4 = vcombine_u64(a2.val[1],  a2.val[3]);
                uint8x16_t p5 = vcombine_u64(a3.val[0] ^ b2.val[0], a3.val[2] ^ b2.val[1]);
                uint8x16_t p6 = vcombine_u64(a3.val[1],  a3.val[3]);
                uint8x16_t p7 = vcombine_u64(a4.val[0] ^ b2.val[2], a4.val[2] ^ b2.val[3]);
                uint8x16_t p8 = vcombine_u64(a4.val[1],  a4.val[3]);

                //t0 = vaddq_u8(t0, vcntq_u8(input.val[0]));
                //t1 = vaddq_u8(t1, vcntq_u8(input.val[1]));
                //t2 = vaddq_u8(t2, vcntq_u8(input.val[2]));
                //t3 = vaddq_u8(t3, vcntq_u8(input.val[3]));

                t0 = vaddq_u8(t0, vcntq_u8(p2));
                t1 = vaddq_u8(t1, vcntq_u8(p4));
                t2 = vaddq_u8(t2, vcntq_u8(p6));
                t3 = vaddq_u8(t3, vcntq_u8(p8));
                r0 = vaddq_u8(r0, vcntq_u8(p1 & p2));
                r1 = vaddq_u8(r1, vcntq_u8(p3 & p4));
                r2 = vaddq_u8(r2, vcntq_u8(p5 & p6));
                r3 = vaddq_u8(r3, vcntq_u8(p7 & p8));
            }
            sum = vpadalq(sum, t0);
            sum = vpadalq(sum, t1);
            sum = vpadalq(sum, t2);
            sum = vpadalq(sum, t3);

            neg = vpadalq(neg, r0);
            neg = vpadalq(neg, r1);
            neg = vpadalq(neg, r2);
            neg = vpadalq(neg, r3);
        } while (i < iters);

        i = 0;
        size %= chunk_size;

        uint64_t tmp[2];
        vst1q_u64(tmp, sum);
        cnt += tmp[0];
        cnt += tmp[1];
        vst1q_u64(tmp, neg);
        cnt = cnt - tmp[0] - tmp[0];
        cnt = cnt - tmp[1] - tmp[1];
    }

    /* Now there can only be 0-7 elements left. */
    // deal with two elements
    uint8x16_t t0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint8x16_t r0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    uint64x2_t neg = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    while (size > 1) {
        uint64x1x4_t a1 = vld4_u64(ptra);
        ptra += load_size;
        uint64x1x4_t b1 = vld4_u64(ptrb);
        ptrb += 2;
        uint8x16_t p1 = vcombine_u64(a1.val[0] ^ b1.val[0], a1.val[2] ^ b1.val[1]);
        uint8x16_t p2 = vcombine_u64(a1.val[1], a1.val[3]);
        t0 = vaddq_u8(t0, vcntq_u8(p2));
        r0 = vaddq_u8(r0, vcntq_u8(p1 & p2));      
        size = size - 2;
    }
    sum = vpadalq(sum, t0);
    neg = vpadalq(neg, r0);
    uint64_t tmp[2];
    vst1q_u64(tmp, sum);
    cnt += tmp[0];
    cnt += tmp[1];
    vst1q_u64(tmp, neg);
    cnt = cnt - tmp[0] - tmp[0];
    cnt = cnt - tmp[1] - tmp[1];

    // deal with one element
    if (size > 0)
    {
        uint64_t p1 = ptra[0] ^ ptrb[0];
        uint64_t p2 = ptra[1];
        uint64_t nega = popcnt64(p1 & p2);
        cnt += popcnt64(p2) - nega - nega;
    }

    return cnt;
}


static inline uint64_t BTNpopKernel(const void* a, const void* b, uint64_t SIZE)
{
    uint64_t i = 0;
    uint64_t size = SIZE;
    uint64_t cnt = 0;
    uint64_t chunk_size = 8;
    //uint64_t chunk_size = 64;
    //const uint8_t* ptr = (const uint8_t*) data;

    uint64_t load_size = 4;
    const uint64_t* ptra = (const uint64_t*)a;
    const uint64_t* ptrb = (const uint64_t*)b;

    if (size >= chunk_size)
    {
        uint64_t iters = size / chunk_size;
        uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint8x16_t zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        do
        {
            uint8x16_t t0 = zero;
            uint8x16_t t1 = zero;
            uint8x16_t t2 = zero;
            uint8x16_t t3 = zero;
            /*
             * After every 31 iterations we need to add the
             * temporary sums (t0, t1, t2, t3) to the total sum.
             * We must ensure that the temporary sums <= 255
             * and 31 * 8 bits = 248 which is OK.
             */
            uint64_t limit = (i + 31 < iters) ? i + 31 : iters;

            /* Each iteration processes 64 bytes */
            for (; i < limit; i++)
            {
                //uint8x16x4_t input = vld4q_u8(ptr);
                //ptr += chunk_size;

                uint64x1x4_t a1 = vld4_u64(ptra);
                ptra += load_size;
                uint64x1x4_t a2 = vld4_u64(ptra);
                ptra += load_size;

                uint64x1x4_t b1 = vld4_u64(ptrb);
                ptrb += load_size;
                uint64x1x4_t b2 = vld4_u64(ptrb);
                ptrb += load_size;
                uint64x1x4_t b3 = vld4_u64(ptrb);
                ptrb += load_size;
                uint64x1x4_t b4 = vld4_u64(ptrb);
                ptrb += load_size;

                uint8x16_t p1 = vcombine_u64((a1.val[0] ^ b1.val[0]) & b1.val[1], (a1.val[1] ^ b1.val[2]) & b1.val[3]);
                uint8x16_t p2 = vcombine_u64((a1.val[2] ^ b2.val[0]) & b2.val[1], (a1.val[3] ^ b2.val[2]) & b2.val[3]);
                uint8x16_t p3 = vcombine_u64((a2.val[0] ^ b3.val[0]) & b3.val[1], (a2.val[1] ^ b3.val[2]) & b3.val[3]);
                uint8x16_t p4 = vcombine_u64((a2.val[2] ^ b4.val[0]) & b4.val[1], (a2.val[3] ^ b4.val[2]) & b4.val[3]);

                t0 = vaddq_u8(t0, vcntq_u8(p1));
                t1 = vaddq_u8(t1, vcntq_u8(p2));
                t2 = vaddq_u8(t2, vcntq_u8(p3));
                t3 = vaddq_u8(t3, vcntq_u8(p4));
            }
            sum = vpadalq(sum, t0);
            sum = vpadalq(sum, t1);
            sum = vpadalq(sum, t2);
            sum = vpadalq(sum, t3);

        } while (i < iters);

        i = 0;
        size %= chunk_size;

        uint64_t tmp[2];
        vst1q_u64(tmp, sum);
        cnt += tmp[0];
        cnt += tmp[1];
    }

    /* Now there can only be 0-7 elements left. */
    // deal with two elements
    uint8x16_t t0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    while (size > 1) {
        uint64x1x4_t a1 = vld4_u64(ptra);
        ptra += 2;
        uint64x1x4_t b1 = vld4_u64(ptrb);
        ptrb += load_size;
        uint8x16_t p1 = vcombine_u64((a1.val[0] ^ b1.val[0]) & b1.val[1], (a1.val[1] ^ b1.val[2]) & b1.val[3]);
        t0 = vaddq_u8(t0, vcntq_u8(p1));
        size = size - 2;
    }
    sum = vpadalq(sum, t0);
    uint64_t tmp[2];
    vst1q_u64(tmp, sum);
    cnt += tmp[0];
    cnt += tmp[1];

    // deal with one element
    if (size > 0)
    {
        cnt += popcnt64((ptra[0] ^ ptrb[0]) & ptrb[1]);
    }

    return cnt;
}


static inline uint64_t BNNpopKernel(const void* a, const void* b, uint64_t SIZE)
{
    uint64_t i = 0;
    uint64_t size = SIZE;
    uint64_t cnt = 0;
    uint64_t chunk_size = 8;
    //uint64_t chunk_size = 64;
    //const uint8_t* ptr = (const uint8_t*) data;

    uint64_t load_size = 4;
    const uint64_t* ptra = (const uint64_t*)a;
    const uint64_t* ptrb = (const uint64_t*)b;

    if (size >= chunk_size)
    {
        uint64_t iters = size / chunk_size;
        uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint8x16_t zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        do
        {
            uint8x16_t t0 = zero;
            uint8x16_t t1 = zero;
            uint8x16_t t2 = zero;
            uint8x16_t t3 = zero;
            /*
             * After every 31 iterations we need to add the
             * temporary sums (t0, t1, t2, t3) to the total sum.
             * We must ensure that the temporary sums <= 255
             * and 31 * 8 bits = 248 which is OK.
             */
            uint64_t limit = (i + 31 < iters) ? i + 31 : iters;

            /* Each iteration processes 64 bytes */
            for (; i < limit; i++)
            {
                //uint8x16x4_t input = vld4q_u8(ptr);
                //ptr += chunk_size;

                uint64x1x4_t a1 = vld4_u64(ptra);
                ptra += load_size;
                uint64x1x4_t a2 = vld4_u64(ptra);
                ptra += load_size;

                uint64x1x4_t b1 = vld4_u64(ptrb);
                ptrb += load_size;
                uint64x1x4_t b2 = vld4_u64(ptrb);
                ptrb += load_size;

                uint8x16_t p1 = vcombine_u64(a1.val[0] ^ b1.val[0], a1.val[1] ^ b1.val[1]);
                uint8x16_t p2 = vcombine_u64(a1.val[2] ^ b1.val[2], a1.val[3] ^ b1.val[3]);
                uint8x16_t p3 = vcombine_u64(a2.val[0] ^ b2.val[0], a2.val[1] ^ b2.val[1]);
                uint8x16_t p4 = vcombine_u64(a2.val[2] ^ b2.val[2], a2.val[3] ^ b2.val[3]);

                t0 = vaddq_u8(t0, vcntq_u8(p1));
                t1 = vaddq_u8(t1, vcntq_u8(p2));
                t2 = vaddq_u8(t2, vcntq_u8(p3));
                t3 = vaddq_u8(t3, vcntq_u8(p4));
            }
            sum = vpadalq(sum, t0);
            sum = vpadalq(sum, t1);
            sum = vpadalq(sum, t2);
            sum = vpadalq(sum, t3);

        } while (i < iters);

        i = 0;
        size %= chunk_size;

        uint64_t tmp[2];
        vst1q_u64(tmp, sum);
        cnt += tmp[0];
        cnt += tmp[1];
    }

    /* Now there can only be 0-7 elements left. */
    // deal with two elements
    uint8x16_t t0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    while (size > 3) {
        uint64x1x4_t a1 = vld4_u64(ptra);
        ptra += load_size;
        uint64x1x4_t b1 = vld4_u64(ptrb);
        ptrb += load_size;
        uint8x16_t p1 = vcombine_u64(a1.val[0] ^ b1.val[0], a1.val[1] ^ b1.val[1]);
        uint8x16_t p2 = vcombine_u64(a1.val[2] ^ b1.val[2], a1.val[3] ^ b1.val[3]);
        t0 = vaddq_u8(t0, vcntq_u8(p1));
        t0 = vaddq_u8(t0, vcntq_u8(p2));
        size = size - load_size;
    }
    sum = vpadalq(sum, t0);
    uint64_t tmp[2];
    vst1q_u64(tmp, sum);
    cnt += tmp[0];
    cnt += tmp[1];

    // deal with one element
    while (size > 0)
    {
        cnt += popcnt64(ptra[0] ^ ptrb[0]);
        ptra++;
        ptrb++;
        size=size-1;
    }

    return cnt;
}


static inline uint64_t DRFoldKernel(const void* a, const void* b, uint64_t SIZE)
{
    uint64_t i = 0;
    uint64_t size = SIZE;
    uint64_t cnt = 0;
    uint64_t chunk_size = 4;
    const int64_t MASK_BIT0 = 0xaaaaaaaaaaaaaaaa;
    const int64_t MASK_BIT1 = 0x5555555555555555;
    //uint64_t chunk_size = 64;
    //const uint8_t* ptr = (const uint8_t*) data;

    uint64_t load_size = 4;
    const uint64_t* ptra = (const uint64_t*)a;
    const uint64_t* ptrb = (const uint64_t*)b;

    if (size >= chunk_size)
    {
        uint64_t iters = size / chunk_size;
        uint64x2_t one = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint64x2_t two = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint64x2_t fou = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint8x16_t zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        do
        {
            uint8x16_t t0 = zero;
            uint8x16_t t1 = zero;
            uint8x16_t t2 = zero;
            uint8x16_t t3 = zero;
            uint8x16_t r0 = zero;
            uint8x16_t r1 = zero;
            uint8x16_t r2 = zero;
            uint8x16_t r3 = zero;
            /*
             * After every 31 iterations we need to add the
             * temporary sums (t0, t1, t2, t3) to the total sum.
             * We must ensure that the temporary sums <= 255
             * and 31 * 8 bits = 248 which is OK.
             */
            uint64_t limit = (i + 31 < iters) ? i + 31 : iters;

            /* Each iteration processes 64 bytes */
            for (; i < limit; i++)
            {
                //uint8x16x4_t input = vld4q_u8(ptr);
                //ptr += chunk_size;

                uint64x1x4_t a1 = vld4_u64(ptra);
                ptra += load_size;

                uint64x1x4_t b1 = vld4_u64(ptrb);
                ptrb += load_size;

                /*
                p1[iw] = (a[(oh * K + iw)] & b[(ow * K + iw)]) & MASK_BIT0;
                p2[iw] = (a[(oh * K + iw)] & (b[(ow * K + iw)] >> 1)) & MASK_BIT1;
                p3[iw] = ((a[(oh * K + iw)] >> 1) & b[(ow * K + iw)]) & MASK_BIT1;
                p4[iw] = (a[(oh * K + iw)] & b[(ow * K + iw)] & MASK_BIT1);
                */

                uint8x16_t p1 = vcombine_u64((a1.val[0] & b1.val[0]) & MASK_BIT0, (a1.val[1] & b1.val[1]) & MASK_BIT0);
                uint8x16_t p2 = vcombine_u64((a1.val[0] & (b1.val[0] >> 1)) & MASK_BIT1, (a1.val[1] & (b1.val[1] >> 1)) & MASK_BIT1);
                uint8x16_t p3 = vcombine_u64(((a1.val[0] >> 1) & b1.val[0]) & MASK_BIT1, ((a1.val[1] >> 1) & b1.val[1]) & MASK_BIT1);
                uint8x16_t p4 = vcombine_u64((a1.val[0] & b1.val[0]) & MASK_BIT1, (a1.val[1] & b1.val[1]) & MASK_BIT1);

                uint8x16_t p5 = vcombine_u64((a1.val[2] & b1.val[2]) & MASK_BIT0, (a1.val[3] & b1.val[3]) & MASK_BIT0);
                uint8x16_t p6 = vcombine_u64((a1.val[2] & (b1.val[2] >> 1)) & MASK_BIT1, (a1.val[3] & (b1.val[3] >> 1)) & MASK_BIT1);
                uint8x16_t p7 = vcombine_u64(((a1.val[2] >> 1) & b1.val[2]) & MASK_BIT1, ((a1.val[3] >> 1) & b1.val[3]) & MASK_BIT1);
                uint8x16_t p8 = vcombine_u64((a1.val[2] & b1.val[2]) & MASK_BIT1, (a1.val[3] & b1.val[3]) & MASK_BIT1);

                //t0 = vaddq_u8(t0, vcntq_u8(input.val[0]));
                //t1 = vaddq_u8(t1, vcntq_u8(input.val[1]));
                //t2 = vaddq_u8(t2, vcntq_u8(input.val[2]));
                //t3 = vaddq_u8(t3, vcntq_u8(input.val[3]));

                t0 = vaddq_u8(t0, vcntq_u8(p1));
                t1 = vaddq_u8(t1, vcntq_u8(p2));
                t2 = vaddq_u8(t2, vcntq_u8(p3));
                t3 = vaddq_u8(t3, vcntq_u8(p4));
                r0 = vaddq_u8(r0, vcntq_u8(p5));
                r1 = vaddq_u8(r1, vcntq_u8(p6));
                r2 = vaddq_u8(r2, vcntq_u8(p7));
                r3 = vaddq_u8(r3, vcntq_u8(p8));
            }
            fou = vpadalq(fou, t0);
            two = vpadalq(two, t1);
            two = vpadalq(two, t2);
            one = vpadalq(one, t3);

            fou = vpadalq(fou, r0);
            two = vpadalq(two, r1);
            two = vpadalq(two, r2);
            one = vpadalq(one, r3);
        } while (i < iters);

        i = 0;
        size %= chunk_size;

        uint64_t tmp[2];
        vst1q_u64(tmp, fou);
        cnt += tmp[0] * 4;
        cnt += tmp[1] * 4;
        vst1q_u64(tmp, two);
        cnt += tmp[0] * 2;
        cnt += tmp[1] * 2;
        vst1q_u64(tmp, one);
        cnt += tmp[0];
        cnt += tmp[1];
    }

    // deal with one element
    while (size > 0)
    {
        uint64_t p1 = (ptra[0] & ptrb[0]) & MASK_BIT0;
        uint64_t p2 = (ptra[0] & (ptrb[1] >> 1)) & MASK_BIT1;
        uint64_t p3 = ((ptra[1] >> 1) & ptrb[0]) & MASK_BIT1;
        uint64_t p4 = (ptra[1] & ptrb[1]) & MASK_BIT1;
        cnt += popcnt64(p1) * 4 + (popcnt64(p2) + popcnt64(p3)) * 2 + popcnt64(p4);
        size = size - 1;
        ptra++;
        ptrb++;
    }

    return cnt;
}


static inline uint64_t DRFpopKernel(const void* a, const void* b, uint64_t SIZE)
{
    uint64_t i = 0;
    uint64_t size = SIZE;
    uint64_t cnt = 0;
    uint64_t chunk_size = 4;
    //uint64_t chunk_size = 64;
    //const uint8_t* ptr = (const uint8_t*) data;

    uint64_t load_size = 4;
    const uint64_t* ptra = (const uint64_t*)a;
    const uint64_t* ptrb = (const uint64_t*)b;

    if (size >= chunk_size)
    {
        uint64_t iters = size / chunk_size;
        uint64x2_t one = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint64x2_t two = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint64x2_t fou = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint8x16_t zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        do
        {
            uint8x16_t t0 = zero;
            uint8x16_t t1 = zero;
            uint8x16_t t2 = zero;
            uint8x16_t t3 = zero;
            uint8x16_t r0 = zero;
            uint8x16_t r1 = zero;
            uint8x16_t r2 = zero;
            uint8x16_t r3 = zero;
            /*
             * After every 31 iterations we need to add the
             * temporary sums (t0, t1, t2, t3) to the total sum.
             * We must ensure that the temporary sums <= 255
             * and 31 * 8 bits = 248 which is OK.
             */
            uint64_t limit = (i + 31 < iters) ? i + 31 : iters;

            /* Each iteration processes 64 bytes */
            for (; i < limit; i++)
            {
                //uint8x16x4_t input = vld4q_u8(ptr);
                //ptr += chunk_size;

                uint64x1x4_t a1 = vld4_u64(ptra);
                ptra += load_size;
                uint64x1x4_t a2 = vld4_u64(ptra);
                ptra += load_size;

                uint64x1x4_t b1 = vld4_u64(ptrb);
                ptrb += load_size;
                uint64x1x4_t b2 = vld4_u64(ptrb);
                ptrb += load_size;

                uint8x16_t p1 = vcombine_u64(a1.val[0] & b1.val[0], a1.val[2] & b1.val[2]);
                uint8x16_t p2 = vcombine_u64(a1.val[0] & b1.val[1], a1.val[1] & b1.val[0]);
                uint8x16_t p3 = vcombine_u64(a1.val[2] & b1.val[3], a1.val[3] & b1.val[2]);
                uint8x16_t p4 = vcombine_u64(a1.val[1] & b1.val[1], a1.val[3] & b1.val[3]);

                uint8x16_t p5 = vcombine_u64(a2.val[0] & b2.val[0], a2.val[2] & b2.val[2]);
                uint8x16_t p6 = vcombine_u64(a2.val[0] & b2.val[1], a2.val[1] & b2.val[0]);
                uint8x16_t p7 = vcombine_u64(a2.val[2] & b2.val[3], a2.val[3] & b2.val[2]);
                uint8x16_t p8 = vcombine_u64(a2.val[1] & b2.val[1], a2.val[3] & b2.val[3]);

                //t0 = vaddq_u8(t0, vcntq_u8(input.val[0]));
                //t1 = vaddq_u8(t1, vcntq_u8(input.val[1]));
                //t2 = vaddq_u8(t2, vcntq_u8(input.val[2]));
                //t3 = vaddq_u8(t3, vcntq_u8(input.val[3]));

                t0 = vaddq_u8(t0, vcntq_u8(p1));
                t1 = vaddq_u8(t1, vcntq_u8(p2));
                t2 = vaddq_u8(t2, vcntq_u8(p3));
                t3 = vaddq_u8(t3, vcntq_u8(p4));
                r0 = vaddq_u8(r0, vcntq_u8(p5));
                r1 = vaddq_u8(r1, vcntq_u8(p6));
                r2 = vaddq_u8(r2, vcntq_u8(p7));
                r3 = vaddq_u8(r3, vcntq_u8(p8));
            }
            fou = vpadalq(fou, t0);
            two = vpadalq(two, t1);
            two = vpadalq(two, t2);
            one = vpadalq(one, t3);

            fou = vpadalq(fou, r0);
            two = vpadalq(two, r1);
            two = vpadalq(two, r2);
            one = vpadalq(one, r3);
        } while (i < iters);

        i = 0;
        size %= chunk_size;

        uint64_t tmp[2];
        vst1q_u64(tmp, fou);
        cnt += tmp[0] * 4;
        cnt += tmp[1] * 4;
        vst1q_u64(tmp, two);
        cnt += tmp[0] * 2;
        cnt += tmp[1] * 2;
        vst1q_u64(tmp, one);
        cnt += tmp[0];
        cnt += tmp[1];
    }

    /* Now there can only be 0, 1, 2, 3 elements left. */
    // deal with two elements
    uint8x16_t t0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint8x16_t t1 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint8x16_t t2 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint8x16_t t3 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint64x2_t fou = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    uint64x2_t two = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    uint64x2_t one = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    while (size > 1) {
        uint64x1x4_t a1 = vld4_u64(ptra);
        ptra += load_size;
        uint64x1x4_t b1 = vld4_u64(ptrb);
        ptrb += load_size;
        uint8x16_t p1 = vcombine_u64(a1.val[0] & b1.val[0], a1.val[2] & b1.val[2]);
        uint8x16_t p2 = vcombine_u64(a1.val[0] & b1.val[1], a1.val[1] & b1.val[0]);
        uint8x16_t p3 = vcombine_u64(a1.val[2] & b1.val[3], a1.val[3] & b1.val[2]);
        uint8x16_t p4 = vcombine_u64(a1.val[1] & b1.val[1], a1.val[3] & b1.val[3]);
        t0 = vaddq_u8(t0, vcntq_u8(p1));
        t1 = vaddq_u8(t1, vcntq_u8(p2));
        t2 = vaddq_u8(t2, vcntq_u8(p3));
        t3 = vaddq_u8(t3, vcntq_u8(p4));     
        size = size - 2;
    }
    fou = vpadalq(fou, t0);
    two = vpadalq(two, t1);
    two = vpadalq(two, t2);
    one = vpadalq(one, t3);
    uint64_t tmp[2];
    vst1q_u64(tmp, fou);
    cnt += tmp[0] * 4;
    cnt += tmp[1] * 4;
    vst1q_u64(tmp, two);
    cnt += tmp[0] * 2;
    cnt += tmp[1] * 2;
    vst1q_u64(tmp, one);
    cnt += tmp[0];
    cnt += tmp[1];

    // deal with one element
    if (size > 0)
    {
        uint64_t p1 = ptra[0] & ptrb[0];
        uint64_t p2 = ptra[0] & ptrb[1];
        uint64_t p3 = ptra[1] & ptrb[0];
        uint64_t p4 = ptra[1] & ptrb[1];
        cnt += popcnt64(p1) * 4 + (popcnt64(p2) + popcnt64(p3)) * 2 + popcnt64(p4);
    }

    return cnt;
}


static inline uint64_t RTNpopKernel(const void* a, const void* b, uint64_t SIZE)
{
    uint64_t i = 0;
    uint64_t size = SIZE;
    uint64_t cnt = 0;
    uint64_t chunk_size = 8;
    const int64_t mask = 0x5555555555555555;
    //uint64_t chunk_size = 64;
    //const uint8_t* ptr = (const uint8_t*) data;

    uint64_t load_size = 4;
    const uint64_t* ptra = (const uint64_t*)a;
    const uint64_t* ptrb = (const uint64_t*)b;

    if (size >= chunk_size)
    {
        uint64_t iters = size / chunk_size;
        uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint64x2_t neg = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
        uint8x16_t zero = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        do
        {
            uint8x16_t t0 = zero;
            uint8x16_t t1 = zero;
            uint8x16_t t2 = zero;
            uint8x16_t t3 = zero;
            uint8x16_t r0 = zero;
            uint8x16_t r1 = zero;
            uint8x16_t r2 = zero;
            uint8x16_t r3 = zero;
            /*
             * After every 31 iterations we need to add the
             * temporary sums (t0, t1, t2, t3) to the total sum.
             * We must ensure that the temporary sums <= 255
             * and 31 * 8 bits = 248 which is OK.
             */
            uint64_t limit = (i + 31 < iters) ? i + 31 : iters;

            /* Each iteration processes 64 bytes */
            for (; i < limit; i++)
            {
                //uint8x16x4_t input = vld4q_u8(ptr);
                //ptr += chunk_size;

                uint64x1x4_t a1 = vld4_u64(ptra);
                ptra += load_size;
                uint64x1x4_t a2 = vld4_u64(ptra);
                ptra += load_size;

                uint64x1x4_t b1 = vld4_u64(ptrb);
                ptrb += load_size;
                uint64x1x4_t b2 = vld4_u64(ptrb);
                ptrb += load_size;

				uint8x16_t p1 = vcombine_u64((a1.val[0] ^ b1.val[0]) >> 1, (a1.val[1] ^ b1.val[1]) >> 1);
				uint8x16_t p2 = vcombine_u64(a1.val[0] & b1.val[0] & mask, a1.val[1] & b1.val[1] & mask);
				uint8x16_t p3 = vcombine_u64((a1.val[2] ^ b1.val[2]) >> 1, (a1.val[3] ^ b1.val[3]) >> 1);
				uint8x16_t p4 = vcombine_u64(a1.val[2] & b1.val[2] & mask, a1.val[3] & b1.val[3] & mask);
				uint8x16_t p5 = vcombine_u64((a2.val[0] ^ b2.val[0]) >> 1, (a2.val[1] ^ b2.val[1]) >> 1);
				uint8x16_t p6 = vcombine_u64(a2.val[0] & b2.val[0] & mask, a2.val[1] & b2.val[1] & mask);
				uint8x16_t p7 = vcombine_u64((a2.val[2] ^ b2.val[2]) >> 1, (a2.val[3] ^ b2.val[3]) >> 1);
				uint8x16_t p8 = vcombine_u64(a2.val[2] & b2.val[2] & mask, a2.val[3] & b2.val[3] & mask);

                //t0 = vaddq_u8(t0, vcntq_u8(input.val[0]));
                //t1 = vaddq_u8(t1, vcntq_u8(input.val[1]));
                //t2 = vaddq_u8(t2, vcntq_u8(input.val[2]));
                //t3 = vaddq_u8(t3, vcntq_u8(input.val[3]));

                t0 = vaddq_u8(t0, vcntq_u8(p2));
                t1 = vaddq_u8(t1, vcntq_u8(p4));
                t2 = vaddq_u8(t2, vcntq_u8(p6));
                t3 = vaddq_u8(t3, vcntq_u8(p8));
                r0 = vaddq_u8(r0, vcntq_u8(p1 & p2));
                r1 = vaddq_u8(r1, vcntq_u8(p3 & p4));
                r2 = vaddq_u8(r2, vcntq_u8(p5 & p6));
                r3 = vaddq_u8(r3, vcntq_u8(p7 & p8));
            }
            sum = vpadalq(sum, t0);
            sum = vpadalq(sum, t1);
            sum = vpadalq(sum, t2);
            sum = vpadalq(sum, t3);

            neg = vpadalq(neg, r0);
            neg = vpadalq(neg, r1);
            neg = vpadalq(neg, r2);
            neg = vpadalq(neg, r3);
        } while (i < iters);

        i = 0;
        size %= chunk_size;

        uint64_t tmp[2];
        vst1q_u64(tmp, sum);
        cnt += tmp[0];
        cnt += tmp[1];
        vst1q_u64(tmp, neg);
        cnt = cnt - tmp[0] - tmp[0];
        cnt = cnt - tmp[1] - tmp[1];
    }

    /* Now there can only be 0-7 elements left. */
    // deal with four elements
    uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    uint64x2_t neg = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
    uint8x16_t t0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint8x16_t r0 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint8x16_t t1 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    uint8x16_t r1 = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    while (size > 3) {
        uint64x1x4_t a1 = vld4_u64(ptra);
        ptra += load_size;
        uint64x1x4_t b1 = vld4_u64(ptrb);
        ptrb += load_size;
        uint8x16_t p1 = vcombine_u64((a1.val[0] ^ b1.val[0]) >> 1, (a1.val[1] ^ b1.val[1]) >> 1);
        uint8x16_t p2 = vcombine_u64(a1.val[0] & b1.val[0] & mask, a1.val[1] & b1.val[1] & mask);
        uint8x16_t p3 = vcombine_u64((a1.val[2] ^ b1.val[2]) >> 1, (a1.val[3] ^ b1.val[3]) >> 1);
        uint8x16_t p4 = vcombine_u64(a1.val[2] & b1.val[2] & mask, a1.val[3] & b1.val[3] & mask);
        t0 = vaddq_u8(t0, vcntq_u8(p2));
        t1 = vaddq_u8(t1, vcntq_u8(p4));
        r0 = vaddq_u8(r0, vcntq_u8(p1 & p2));
        r1 = vaddq_u8(r1, vcntq_u8(p3 & p4));
        size = size - load_size;
    }
    sum = vpadalq(sum, t0);
    neg = vpadalq(neg, r0);
    sum = vpadalq(sum, t1);
    neg = vpadalq(neg, r1);
    uint64_t tmp[2];
    vst1q_u64(tmp, sum);
    cnt += tmp[0];
    cnt += tmp[1];
    vst1q_u64(tmp, neg);
    cnt = cnt - tmp[0] - tmp[0];
    cnt = cnt - tmp[1] - tmp[1];

    // deal with one element
    while(size > 0)
    {
        uint64_t p1 = (ptra[0] ^ ptrb[0]) >> 1;
        uint64_t p2 = ptra[1] & ptrb[1] & mask;
        uint64_t nega = popcnt64(p1 & p2);
        cnt += popcnt64(p2) - nega - nega;
        ptra++;
        ptrb++;
        size=size-1;
    }

    return cnt;
}



#endif

#endif /* LIBPOPCNT_H */
