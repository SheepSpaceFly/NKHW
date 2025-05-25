#pragma once
#include <arm_neon.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <queue>

std::priority_queue<std::pair<float, uint32_t> > pq_search(uint8_t* compressed_base, float *centers, float *centers_dis, float* query, size_t base_number, size_t vecdim, size_t k_top) {
    const int K=256;
    // int query_labels[4];
    float query_centers_dis[4][256];

    for(int i=0;i<4;i++){
        float min_dis = INFINITY;
        for(int j=0;j<256;j++){
            float *center = centers+i*K*vecdim/4+j*vecdim/4;
            float *sub_query = query+i*vecdim/4;
            float32x4_t sum_vec = vdupq_n_f32(0);
            for(int k=0;k<vecdim/4;k+=4){
                float32x4_t vec_1 = vld1q_f32(center+k);
                float32x4_t vec_2 = vld1q_f32(sub_query+k);
                float32x4_t diff = vsubq_f32(vec_1, vec_2);
                float32x4_t squared_diff  = vmulq_f32(diff,diff);
                sum_vec = vaddq_f32(sum_vec,squared_diff);
            }
            float sum = vaddvq_f32(sum_vec);
            // if(dis<min_dis){
            //     min_dis = dis;
            //     query_labels[i] = j;
            // }
            query_centers_dis[i][j] = sum;
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < base_number; i++) {
        float dis = 0;
        for(int j=0;j<4;j++){
            // dis+=centers_dis[j*256*256+compressed_base[i*4+j]*256+query_labels[j]];
            dis+=query_centers_dis[j][compressed_base[i*4+j]];
        }
        if(q.size() < k_top) {
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

struct pq_thread_parm{
    uint8_t* compressed_base;
    float *centers;
    float *centers_dis;
    float* query;
    size_t base_number;
    size_t vecdim;
    size_t k_top;
    pthread_mutex_t *amutex;
    int thread_number;
    int thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *main_q;
    float *query_centers_dis;
}

void* query_centers_dis_thread(void* _parm){
    float *centers = parm->centers;
    float *centers_dis = parm->centers_dis;
    float* query = parm->query;
    size_t vecdim = parm->vecdim;
    int thread_number = parm->thread_number;
    int thread_id = parm->thread_id;
    float *query_centers_dis = parm->query_centers_dis;

    int my_first = 4*256/thread_number*thread_id;
    int my_last = std::min(my_first+4*256/thread_number,4*256)

    for(int i=my_first;i<my_last;i++){
        float *center = centers+i*K*vecdim/4+j*vecdim/4;
        float *sub_query = query+i*vecdim/4;
    }


    for(int i=0;i<4;i++){
        float min_dis = INFINITY;
        for(int j=0;j<256;j++){
            float *center = centers+i*K*vecdim/4+j*vecdim/4;
            float *sub_query = query+i*vecdim/4;
            float32x4_t sum_vec = vdupq_n_f32(0);
            for(int k=0;k<vecdim/4;k+=4){
                float32x4_t vec_1 = vld1q_f32(center+k);
                float32x4_t vec_2 = vld1q_f32(sub_query+k);
                float32x4_t diff = vsubq_f32(vec_1, vec_2);
                float32x4_t squared_diff  = vmulq_f32(diff,diff);
                sum_vec = vaddq_f32(sum_vec,squared_diff);
            }
            float sum = vaddvq_f32(sum_vec);
            query_centers_dis[i][j] = sum;
        }
    }
}

void* sq_search_thread(void* _parm){
    thread_parm* parm = (thread_parm*)_parm;
    uint8_t* compressed_base = parm->compressed_base;
    float *centers = parm->centers;
    float *centers_dis = parm->centers_dis;
    float* query = parm->query;
    size_t base_number = parm->base_number;
    size_t vecdim = parm->vecdim;
    size_t k_top = parm->k_top;
    pthread_mutex_t *amutex = parm->amutex;
    int thread_number = parm->thread_number;
    int thread_id = parm->thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *main_q = parm->main_q;
    float *query_centers_dis = parm->query_centers_dis;

    const int K=256;

    for(int i=0;i<4;i++){
        float min_dis = INFINITY;
        for(int j=0;j<256;j++){
            float *center = centers+i*K*vecdim/4+j*vecdim/4;
            float *sub_query = query+i*vecdim/4;
            float32x4_t sum_vec = vdupq_n_f32(0);
            for(int k=0;k<vecdim/4;k+=4){
                float32x4_t vec_1 = vld1q_f32(center+k);
                float32x4_t vec_2 = vld1q_f32(sub_query+k);
                float32x4_t diff = vsubq_f32(vec_1, vec_2);
                float32x4_t squared_diff  = vmulq_f32(diff,diff);
                sum_vec = vaddq_f32(sum_vec,squared_diff);
            }
            float sum = vaddvq_f32(sum_vec);
            query_centers_dis[i][j] = sum;
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > q;

        for(int i = 0; i < base_number; i++) {
        float dis = 0;
        for(int j=0;j<4;j++){
            dis+=query_centers_dis[j][compressed_base[i*4+j]];
        }
        if(q.size() < k_top) {
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