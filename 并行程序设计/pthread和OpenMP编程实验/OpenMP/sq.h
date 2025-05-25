#pragma once
#include <arm_neon.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <queue>
#include <algorithm>
#include <pthread.h>

void sq_compress_base(float* base, uint8_t* compressed_base, size_t base_number, size_t vecdim){
    int byte_number = base_number*vecdim;

    float32x4_t min_vec = vdupq_n_f32(INFINITY);
    float32x4_t max_vec = vdupq_n_f32(-INFINITY);

    for(int i=0;i+4<=byte_number;i+=4){
        float32x4_t vec = vld1q_f32(base+i);
        min_vec = vminq_f32(min_vec,vec);
        max_vec = vmaxq_f32(max_vec,vec);
    }
    float32x4_t temp1 = vminq_f32(min_vec, vrev64q_f32(min_vec));
    float32x2_t temp2 = vpmin_f32(vget_low_f32(temp1), vget_high_f32(temp1));
    float min_val = vget_lane_f32(temp2, 0);
    temp1 = vmaxq_f32(max_vec, vrev64q_f32(max_vec));
    temp2 = vpmax_f32(vget_low_f32(temp1), vget_high_f32(temp1));
    float max_val = vget_lane_f32(temp2, 0);
    for(int i=(byte_number/4)*4;i<byte_number;i++){
        if(base[i]<min_val) min_val = base[i];
        if(base[i]>max_val) max_val = base[i];
    }
    
    if(min_val==max_val){
        uint8x16_t zero_vec = vdupq_n_u8(0);
        for(int i=0;i<=byte_number-16;i+=16){
            vst1q_u8(compressed_base+i,zero_vec);
        }
        for(int i=(byte_number/16)*16;i<byte_number;i++){
            compressed_base[i] = 0;
        }
        return;
    }

    float scale = 255.0f/(max_val-min_val);
    float32x4_t vmin = vdupq_n_f32(min_val);
    float32x4_t vscale = vdupq_n_f32(scale);

    for(int i=0;i+16<=byte_number;i+=16){
        float32x4_t x0 = vld1q_f32(base+i);
        float32x4_t x1 = vld1q_f32(base+i+4);
        float32x4_t x2 = vld1q_f32(base+i+8);
        float32x4_t x3 = vld1q_f32(base+i+12);

        x0 = vsubq_f32(x0, vmin);
        x1 = vsubq_f32(x1, vmin);
        x2 = vsubq_f32(x2, vmin);
        x3 = vsubq_f32(x3, vmin);
        x0 = vmulq_f32(x0, vscale);
        x1 = vmulq_f32(x1, vscale);
        x2 = vmulq_f32(x2, vscale);
        x3 = vmulq_f32(x3, vscale);
        x0 = vmaxq_f32(vminq_f32(x0, vdupq_n_f32(255.0f)), vdupq_n_f32(0.0f));
        x1 = vmaxq_f32(vminq_f32(x1, vdupq_n_f32(255.0f)), vdupq_n_f32(0.0f));
        x2 = vmaxq_f32(vminq_f32(x2, vdupq_n_f32(255.0f)), vdupq_n_f32(0.0f));
        x3 = vmaxq_f32(vminq_f32(x3, vdupq_n_f32(255.0f)), vdupq_n_f32(0.0f));
        int32x4_t y0 = vcvtnq_s32_f32(x0);
        int32x4_t y1 = vcvtnq_s32_f32(x1);
        int32x4_t y2 = vcvtnq_s32_f32(x2);
        int32x4_t y3 = vcvtnq_s32_f32(x3);
        int16x4_t y0_16 = vmovn_s32(y0);
        int16x4_t y1_16 = vmovn_s32(y1);
        int16x4_t y2_16 = vmovn_s32(y2);
        int16x4_t y3_16 = vmovn_s32(y3);
        int16x8_t y01 = vcombine_s16(y0_16, y1_16);
        int16x8_t y23 = vcombine_s16(y2_16, y3_16);
        uint8x8_t y01_u8 = vqmovun_s16(y01);
        uint8x8_t y23_u8 = vqmovun_s16(y23);
        vst1q_u8(compressed_base+i,vcombine_u8(y01_u8,y23_u8));
    }
    for(int i=(byte_number/16)*16;i<byte_number;i++){
        float x = base[i];
        float y = (x-min_val)*scale;
        y = std::max(0.0f,std::min(y,255.0f));
        compressed_base[i] = static_cast<uint8_t>(y+0.5f);
    }
}

float sq_l2_distance(uint8_t* query1, uint8_t* query2, size_t vecdim){//平方距离
    uint32x4_t sum_vec = vdupq_n_u32(0);
    for(int i=0;i<vecdim;i+=16){
        uint8x16_t x = vld1q_u8(query1+i);
        uint8x16_t y = vld1q_u8(query2+i);
        uint8x16_t diff = vabdq_u8(x,y);
        uint8x8_t diff_lo = vget_low_u8(diff);
        uint8x8_t diff_hi = vget_high_u8(diff);
        uint16x8_t diff2_lo = vmull_u8(diff_lo,diff_lo);
        uint16x8_t diff2_hi = vmull_u8(diff_hi,diff_hi);

        sum_vec = vaddq_u32(sum_vec,vpaddlq_u16(diff2_lo));
        sum_vec = vaddq_u32(sum_vec,vpaddlq_u16(diff2_hi));
    }
    uint32_t sum = vaddvq_u32(sum_vec);
    return sum;
}

struct thread_parm{
    uint8_t* base;
    uint8_t* query;
    size_t base_number;
    size_t vecdim;
    size_t k;
    pthread_mutex_t *amutex;
    int thread_number;
    int thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *main_q;
};

void* sq_search_thread(void* _parm){
    std::priority_queue<std::pair<float, uint32_t> > q;
    thread_parm* parm = (thread_parm*)_parm;
    uint8_t* base = parm->base;
    uint8_t* query = parm->query;
    size_t base_number = parm->base_number;
    size_t vecdim = parm->vecdim;
    size_t k = parm->k;
    pthread_mutex_t *amutex = parm->amutex;
    int thread_number = parm->thread_number;
    int thread_id = parm->thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *main_q = parm->main_q;

    int my_first = thread_id*base_number/thread_number;
    int my_last = std::min(base_number/thread_number + my_first,base_number);
    
    for(int i = my_first; i < my_last ; ++i) {
        float dis = 0;
        uint8_t* base_query = base+i*vecdim;
        
        dis = sq_l2_distance(query,base_query,vecdim);

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.pop();
                q.push({dis, i});
            }
        }
    }
    pthread_mutex_lock(amutex);
    while (q.size()) {   
        if(main_q->size() < k) {
            main_q->push(q.top());
        } else {
            if(q.top().first < main_q->top().first) {
                main_q->pop();
                main_q->push(q.top());
            }
        }
        q.pop();
    }
    pthread_mutex_unlock(amutex);
    return nullptr;
}

std::priority_queue<std::pair<float, uint32_t> > sq_search(uint8_t* base, uint8_t* query, size_t base_number, size_t vecdim, size_t k, int thread_number) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    pthread_t threads[16];
    thread_parm parm[16];
    pthread_mutex_t amutex = PTHREAD_MUTEX_INITIALIZER;

    for(int thread_id=0;thread_id<thread_number;thread_id++){
        parm[thread_id] = {
            base,query,base_number,vecdim,k,&amutex,thread_number,thread_id,&q
        };
        pthread_create(&threads[thread_id],NULL,sq_search_thread,&parm[thread_id]);
    }

    for(int i=0;i<thread_number;i++){
        pthread_join(threads[i],NULL);
    }

    return q;
}

std::priority_queue<std::pair<float, uint32_t> > sq_search_openmp(uint8_t* base, uint8_t* query, size_t base_number, size_t vecdim, size_t k, int thread_number) {
    std::priority_queue<std::pair<float, uint32_t> > main_q;

    #pragma omp parallel num_threads(thread_number)
    {
        std::priority_queue<std::pair<float, uint32_t> > local_q;
        #pragma omp for schedule(static)
        for(int i = 0; i < base_number; ++i) {
            float dis = 0;
            uint8_t* base_query = base+i*vecdim;
            
            dis = sq_l2_distance(query,base_query,vecdim);

        
            if(local_q.size() < k) {
                local_q.push({dis, i});
            } else {
                if(dis < local_q.top().first) {
                    local_q.push({dis, i});
                    local_q.pop();
                }
            }
        }
        #pragma omp critical
        {
            while(!local_q.empty()) {
                auto elem = local_q.top();
                if(main_q.size() < k) {
                    main_q.push(elem);
                } else if(elem.first < main_q.top().first) {
                    main_q.push(elem);
                    main_q.pop();
                }
                local_q.pop();
            }
        }
    }
    return main_q;
}

void sq_save_compressed_base(uint8_t* compressed_base, size_t base_number, size_t vecdim){
    int byte_number = base_number*vecdim;
    std::ofstream out_file("./files/sq_compressed_base.bin",std::ios::binary);
    if(!out_file){
        std::cout<<"无法打开sq_compressed_base.bin文件！"<<std::endl;
        return;
    }

    out_file.write(reinterpret_cast<const char*>(compressed_base), byte_number);

    if(!out_file.good()){
        std::cout<<"写入sq_compressed_base.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入sq_compressed_base.bin！"<<std::endl;
}

void sq_read_compressed_base(uint8_t* compressed_base, size_t base_number, size_t vecdim){
    int byte_number = base_number*vecdim;
    std::ifstream in_file("./files/sq_compressed_base.bin",std::ios::binary);
    if (!in_file.is_open()) {
        std::cout<<"无法打开sq_compressed_base.bin文件！"<<std::endl;
        return;
    }
    if(!in_file.read(reinterpret_cast<char*>(compressed_base),byte_number)){
        std::cout<<"读入sq_compressed_base.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入sq_compressed_base.bin！"<<std::endl;
}



std::priority_queue<std::pair<float, uint32_t> > old_sq_search(uint8_t* base, uint8_t* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < base_number; ++i) {
        float dis = 0;
        uint8_t* base_query = base+i*vecdim;
        
        dis = sq_l2_distance(query,base_query,vecdim);

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
