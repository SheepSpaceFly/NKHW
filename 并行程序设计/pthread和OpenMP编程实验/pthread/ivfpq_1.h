#pragma once
#include <arm_neon.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <queue>
#include <vector>

void build_inverted_kmeans(uint8_t* labels, float* centers, size_t base_number, size_t vecdim, int M){
    std::vector<int>* inverted_kmeans = new std::vector<int>[M];
    for(int i=0;i<base_number;i++){
        inverted_kmeans[labels[i]].push_back(i);
    }

    std::ofstream out_file("./files/inverted_kmeans.bin",std::ios::binary);
    if(!out_file){
        std::cout<<"无法打开inverted_kmeans.bin文件！"<<std::endl;
        return;
    }

    for(int i=0;i<M;i++){
        int size = inverted_kmeans[i].size();
        out_file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        out_file.write(reinterpret_cast<const char*>(inverted_kmeans[i].data()), size*sizeof(int));
    }

    if(!out_file.good()){
        std::cout<<"写入inverted_kmeans.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入inverted_kmeans.bin！"<<std::endl;


    int size = M*vecdim*sizeof(float);

    std::ofstream out_file_2("./files/ivf_centers.bin",std::ios::binary);
    if(!out_file_2){
        std::cout<<"无法打开inverted_kmeans.bin文件！"<<std::endl;
        return;
    }

    out_file_2.write(reinterpret_cast<const char*>(centers), size);

    if(!out_file_2.good()){
        std::cout<<"写入ivf_centers.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入ivf_centers.bin！"<<std::endl;

    size = base_number;

    std::ofstream out_file_3("./files/ivf_base.bin",std::ios::binary);
    if(!out_file_3){
        std::cout<<"无法打开ivf_base.bin文件！"<<std::endl;
        return;
    }

    out_file_3.write(reinterpret_cast<const char*>(labels), size);

    if(!out_file_3.good()){
        std::cout<<"写入ivf_base.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入ivf_base.bin！"<<std::endl;
}

void read_inverted_kmeans(std::vector<int> *inverted_kmeans, float* ivf_centers, uint8_t *ivf_base, size_t base_number, size_t vecdim, int M){
    std::ifstream in_file("./files/inverted_kmeans.bin",std::ios::binary);
    if (!in_file.is_open()) {
        std::cout<<"无法打开inverted_kmeans文件！"<<std::endl;
        return;
    }
    for(int i=0;i<M;i++){
        int size=0;
        if(!in_file.read(reinterpret_cast<char*>(&size),sizeof(int))){
            std::cout<<"读入inverted_kmeans.bin出错！"<<std::endl;
            return;
        }
        int *data = new int[size];
        if(!in_file.read(reinterpret_cast<char*>(data),size*sizeof(int))){
            std::cout<<"读入inverted_kmeans.bin出错！"<<std::endl;
            return;
        }
        for(int j=0;j<size;j++){
            inverted_kmeans[i].push_back(data[j]);
        }
        delete[] data;
    }
    std::cout<<"成功读入inverted_kmeans.bin！"<<std::endl;

    int size = M*vecdim*sizeof(float);
    std::ifstream in_file_2("./files/ivf_centers.bin",std::ios::binary);
    if (!in_file_2.is_open()) {
        std::cout<<"无法打开ivf_centers文件！"<<std::endl;
        return;
    }
    if(!in_file_2.read(reinterpret_cast<char*>(ivf_centers),size)){
        std::cout<<"读入ivf_centers.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入ivf_centers.bin！"<<std::endl;

    size = base_number;
    std::ifstream in_file_3("./files/ivf_base.bin",std::ios::binary);
    if (!in_file_3.is_open()) {
        std::cout<<"无法打开ivf_base文件！"<<std::endl;
        return;
    }
    if(!in_file_3.read(reinterpret_cast<char*>(ivf_base),size)){
        std::cout<<"读入ivf_base.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入ivf_base.bin！"<<std::endl;
}

float get_query_dis(float *query1, float *query2, int vecdim){
    float32x4_t sum_vec = vdupq_n_f32(0);
    for(int k=0;k<vecdim;k+=4){
        float32x4_t vec_1 = vld1q_f32(query1+k);
        float32x4_t vec_2 = vld1q_f32(query2+k);
        float32x4_t diff = vsubq_f32(vec_1, vec_2);
        float32x4_t squared_diff  = vmulq_f32(diff,diff);
        sum_vec = vaddq_f32(sum_vec,squared_diff);
    }
    float sum = vaddvq_f32(sum_vec);
    return sum;
}

struct ivfpq_ivfdis_parm{
    uint8_t* compressed_base;
    float *centers;
    float *ivf_centers;
    std::vector<int> *inverted_kmeans;
    float* query;
    size_t base_number;
    size_t vecdim;
    size_t k_top;
    int m;
    pthread_mutex_t *amutex;
    int thread_number;
    int thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *ivf_q;
    std::priority_queue<std::pair<float, uint32_t> > *my_ivf_q;
};

void* ivfpq_ivfdis_thread(void* _parm){
    ivfpq_ivfdis_parm* parm = (ivfpq_ivfdis_parm*)_parm;
    size_t vecdim = parm->vecdim;
    int m = parm->m;
    float *ivf_centers = parm->ivf_centers;
    float *query = parm->query;
    pthread_mutex_t *amutex = parm->amutex;
    std::priority_queue<std::pair<float, uint32_t> > *ivf_q = parm->ivf_q;
    int thread_number = parm->thread_number;
    int thread_id = parm->thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *my_ivf_q = parm->my_ivf_q;

    int my_first = 256/thread_number*thread_id;
    int my_last = std::min(my_first+256/thread_number,256);

    for(int i = my_first; i < my_last; i++) {
        float dis = get_query_dis(ivf_centers+i*vecdim, query, vecdim);
        if(my_ivf_q->size() < m) {
            my_ivf_q->push({dis, i});
        } else {
            if(dis < my_ivf_q->top().first) {
                my_ivf_q->push({dis, i});
                my_ivf_q->pop();
            }
        }
    }
    return nullptr;
}

struct ivf_pq_parm{
    uint8_t* compressed_base;
    float *centers;
    size_t vecdim;
    size_t k_top;
    int m_i;
    pthread_mutex_t *amutex;
    std::priority_queue<std::pair<float, uint32_t> > *main_q;
    int size;
    int **candidate_base;
    float **residual;
    int thread_number;
    int thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *my_pq_q;
};

void* ivfpq_pq_thread(void* _parm){
    ivf_pq_parm* parm = (ivf_pq_parm*) _parm;
    uint8_t* compressed_base = parm->compressed_base;
    float *centers = parm->centers;
    size_t vecdim = parm->vecdim;
    size_t k_top = parm->k_top;
    int m_i = parm->m_i;
    pthread_mutex_t *amutex = parm->amutex;
    std::priority_queue<std::pair<float, uint32_t> > *main_q = parm->main_q;
    int size = parm->size;
    int **candidate_base = parm->candidate_base;
    float **residual = parm->residual;
    int thread_number = parm->thread_number;
    int thread_id = parm->thread_id;
    std::priority_queue<std::pair<float, uint32_t> > *my_pq_q = parm->my_pq_q;

    int chunk = (size+thread_number-1)/thread_number;
    int my_first = chunk*thread_id;
    int my_last = std::min(my_first+chunk,size);
    
    for(int i=my_first;i<my_last;i++){
        int candidate_query = candidate_base[m_i][i];
        float dis = 0;
        for(int k=0;k<4;k++){
            dis+=get_query_dis(residual[m_i]+k*vecdim/4,centers+k*256*vecdim/4+compressed_base[candidate_query*4+k]*vecdim/4, vecdim/4);
        }
        if(my_pq_q->size() < k_top) {
            my_pq_q->push({dis, candidate_query});
        } else {
            if(dis < my_pq_q->top().first) {
                my_pq_q->push({dis, candidate_query});
                my_pq_q->pop();
            }
        }
    }
    return nullptr;
}

std::priority_queue<std::pair<float, uint32_t> > ivfpq_search(uint8_t* compressed_base, float *centers, float *ivf_centers,std::vector<int> *inverted_kmeans, float* query, size_t base_number, size_t vecdim, size_t k_top, int m, int thread_number) {
    const int K=256;

    int *query_ivf_labels = new int[m];

    std::priority_queue<std::pair<float, uint32_t> > ivf_q;
    std::priority_queue<std::pair<float, uint32_t> > my_q[8];

    pthread_t threads[8];
    ivfpq_ivfdis_parm ivfdis_parm[8];
    pthread_mutex_t amutex = PTHREAD_MUTEX_INITIALIZER;

    for(int thread_id=0;thread_id<thread_number;thread_id++){
        ivfdis_parm[thread_id] = {
            compressed_base,centers,ivf_centers,inverted_kmeans,query,base_number,vecdim,
            k_top,m,&amutex,thread_number,thread_id,&ivf_q,my_q+thread_id
        };
        pthread_create(&threads[thread_id],NULL,ivfpq_ivfdis_thread,&ivfdis_parm[thread_id]);
    }

    for(int i=0;i<thread_number;i++){
        pthread_join(threads[i],NULL);
    }

    for(int i=0;i<thread_number;i++){
        while (my_q[i].size()) {   
            if(ivf_q.size() < m) {
                ivf_q.push(my_q[i].top());
            } else {
                if(my_q[i].top().first < ivf_q.top().first) {
                    ivf_q.pop();
                    ivf_q.push(my_q[i].top());
                }
            }
            my_q[i].pop();
        }
    }

    for(int i=0;i<m;i++){
        query_ivf_labels[i] = ivf_q.top().second;
        ivf_q.pop();
    }

    int *size = new int[m];
    int **candidate_base = new int*[m];
    float **residual = new float*[m];
    for(int i=0; i<m; i++){
        size[i] = inverted_kmeans[query_ivf_labels[i]].size();
        candidate_base[i] = inverted_kmeans[query_ivf_labels[i]].data();
        residual[i] = new float[vecdim];

        float* center_ptr = &ivf_centers[query_ivf_labels[i]*vecdim];
        float* residual_ptr = residual[i];

        for(int j=0; j<vecdim; j+=4){
            float32x4_t center = vld1q_f32(center_ptr + j);
            float32x4_t qvec = vld1q_f32(query + j);
            float32x4_t residual_vec = vsubq_f32(qvec, center);
            vst1q_f32(residual_ptr + j, residual_vec);
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > main_q;
    ivf_pq_parm pq_parm[16];

    for(int i = 0; i < m; i++) {
        for(int thread_id=0;thread_id<thread_number;thread_id++){
            pq_parm[thread_id] = {
                compressed_base,centers,vecdim,k_top,i,&amutex,&main_q,size[i],candidate_base,residual,thread_number,thread_id,my_q+thread_id
            };
            pthread_create(&threads[thread_id],NULL,ivfpq_pq_thread,&pq_parm[thread_id]);
        }
        for(int j=0;j<thread_number;j++){
            pthread_join(threads[j],NULL);
        }
    }

    for(int i=0;i<thread_number;i++){
        while (my_q[i].size()) {   
            if(main_q.size() < k_top) {
                main_q.push(my_q[i].top());
            } else {
                if(my_q[i].top().first < main_q.top().first) {
                    main_q.pop();
                    main_q.push(my_q[i].top());
                }
            }
            my_q[i].pop();
        }
    }

    return main_q;
}

std::priority_queue<std::pair<float, uint32_t> > ivfpq_search_clock(uint8_t* compressed_base, float *centers, float *ivf_centers,std::vector<int> *inverted_kmeans, float* query, size_t base_number, size_t vecdim, size_t k_top, int m, int thread_number,
                                                            double &IVF_avg_time,double &candidate_avg_time,double &search_avg_time) {
    const int K=256;

    int *query_ivf_labels = new int[m];

    std::priority_queue<std::pair<float, uint32_t> > ivf_q;
    std::priority_queue<std::pair<float, uint32_t> > my_q[8];

    pthread_t threads[8];
    ivfpq_ivfdis_parm ivfdis_parm[8];
    pthread_mutex_t amutex = PTHREAD_MUTEX_INITIALIZER;

    auto start_ivf_stage = std::chrono::high_resolution_clock::now();

    for(int thread_id=0;thread_id<thread_number;thread_id++){
        ivfdis_parm[thread_id] = {
            compressed_base,centers,ivf_centers,inverted_kmeans,query,base_number,vecdim,
            k_top,m,&amutex,thread_number,thread_id,&ivf_q,my_q+thread_id
        };
        pthread_create(&threads[thread_id],NULL,ivfpq_ivfdis_thread,&ivfdis_parm[thread_id]);
    }

    for(int i=0;i<thread_number;i++){
        pthread_join(threads[i],NULL);
    }

    for(int i=0;i<thread_number;i++){
        while (my_q[i].size()) {   
            if(ivf_q.size() < m) {
                ivf_q.push(my_q[i].top());
            } else {
                if(my_q[i].top().first < ivf_q.top().first) {
                    ivf_q.pop();
                    ivf_q.push(my_q[i].top());
                }
            }
            my_q[i].pop();
        }
    }

    auto end_ivf_stage = std::chrono::high_resolution_clock::now();
    IVF_avg_time+=std::chrono::duration_cast<std::chrono::microseconds>(end_ivf_stage - start_ivf_stage).count();

    for(int i=0;i<m;i++){
        query_ivf_labels[i] = ivf_q.top().second;
        ivf_q.pop();
    }

    auto start_prepare_candidates = std::chrono::high_resolution_clock::now();

    int* ivf_size_flag = new int[m+1];
    ivf_size_flag[0] = 0;
    for(int i=0;i<m;i++){
        int size = inverted_kmeans[query_ivf_labels[i]].size();
        ivf_size_flag[i+1] = ivf_size_flag[i] + size;
    }
    int size_all = ivf_size_flag[m];

    int *candidate_queries = new int[size_all];
    int *residual_i = new int[size_all];
    float **residual = new float*[m];

    for(int i=0; i<m; i++) {
        int start = ivf_size_flag[i];
        int end = ivf_size_flag[i+1];
        std::vector<int> *candidate_base = &inverted_kmeans[query_ivf_labels[i]];
        for(int j=0; j<end-start; j++) {
            candidate_queries[start+j] = (*candidate_base)[j];
            residual_i[start+j] = i;
        }
        residual[i] = new float[vecdim];
        float* center_ptr = &ivf_centers[query_ivf_labels[i]*vecdim];
        float* residual_ptr = residual[i];
        for(int j=0; j<vecdim; j+=4){
            float32x4_t center = vld1q_f32(center_ptr + j);
            float32x4_t qvec = vld1q_f32(query + j);
            float32x4_t residual_vec = vsubq_f32(qvec, center);
            vst1q_f32(residual_ptr + j, residual_vec);
        }
    }

    auto end_prepare_candidates = std::chrono::high_resolution_clock::now();
    candidate_avg_time+=std::chrono::duration_cast<std::chrono::microseconds>(end_prepare_candidates - start_prepare_candidates).count();

    auto start_search_stage = std::chrono::high_resolution_clock::now();

    std::priority_queue<std::pair<float, uint32_t> > main_q;
    ivf_pq_parm pq_parm[16];

    for(int thread_id=0;thread_id<thread_number;thread_id++){
        pq_parm[thread_id] = {
            compressed_base,centers,vecdim,k_top,size_all,&amutex,&main_q,candidate_queries,residual,
            residual_i,thread_number,thread_id,my_q+thread_id
        };
        pthread_create(&threads[thread_id],NULL,ivfpq_pq_thread,&pq_parm[thread_id]);
    }
    for(int j=0;j<thread_number;j++){
        pthread_join(threads[j],NULL);
    }

    for(int i=0;i<thread_number;i++){
        while (my_q[i].size()) {   
            if(main_q.size() < k_top) {
                main_q.push(my_q[i].top());
            } else {
                if(my_q[i].top().first < main_q.top().first) {
                    main_q.pop();
                    main_q.push(my_q[i].top());
                }
            }
            my_q[i].pop();
        }
    }

    auto end_search_stage = std::chrono::high_resolution_clock::now();
    search_avg_time+=std::chrono::duration_cast<std::chrono::microseconds>(end_search_stage - start_search_stage).count();

    return main_q;
}