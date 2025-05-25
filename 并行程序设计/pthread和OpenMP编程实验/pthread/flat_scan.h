#pragma once
#include <arm_neon.h>
#include <queue>

// float get_query_dis(float *query1, float *query2, int vecdim){
//     float32x4_t sum_vec = vdupq_n_f32(0);
//     for(int k=0;k<vecdim;k+=4){
//         float32x4_t vec_1 = vld1q_f32(query1+k);
//         float32x4_t vec_2 = vld1q_f32(query2+k);
//         float32x4_t mul_vec = vmulq_f32(vec_1,vec_2);
//         sum_vec = vaddq_f32(sum_vec,mul_vec);
//     }
//     float tmp[4];
//     vst1q_f32(tmp,sum_vec);
//     float dis = tmp[0]+tmp[1]+tmp[2]+tmp[3];
//     dis = 1-dis;
//     return dis;
// }


std::priority_queue<std::pair<float, uint32_t> > flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < base_number; ++i) {
        float dis = 0;

        // DEEP100K数据集使用ip距离
        for(int d = 0; d < vecdim; ++d) {
            dis += base[d + i*vecdim]*query[d];
        }
        dis = 1-dis;


        // dis = get_query_dis(base+i*vecdim,query,vecdim);

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}

