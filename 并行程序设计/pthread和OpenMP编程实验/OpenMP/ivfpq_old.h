#pragma once
#include <arm_neon.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <queue>
#include <vector>

void build_inverted_kmeans(uint8_t* labels, float* centers, size_t base_number, size_t vecdim){
    std::vector<int> inverted_kmeans[256];
    for(int i=0;i<base_number;i++){
        inverted_kmeans[labels[i]].push_back(i);
    }

    std::ofstream out_file("./files/inverted_kmeans.bin",std::ios::binary);
    if(!out_file){
        std::cout<<"无法打开inverted_kmeans.bin文件！"<<std::endl;
        return;
    }

    for(int i=0;i<256;i++){
        int size = inverted_kmeans[i].size();
        out_file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        out_file.write(reinterpret_cast<const char*>(inverted_kmeans[i].data()), size*sizeof(int));
    }

    if(!out_file.good()){
        std::cout<<"写入inverted_kmeans.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入inverted_kmeans.bin！"<<std::endl;


    int size = 256*vecdim*sizeof(float);

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
}

void read_inverted_kmeans(std::vector<int> *inverted_kmeans, float* ivf_centers, size_t vecdim){
    std::ifstream in_file("./files/inverted_kmeans.bin",std::ios::binary);
    if (!in_file.is_open()) {
        std::cout<<"无法打开inverted_kmeans文件！"<<std::endl;
        return;
    }
    for(int i=0;i<256;i++){
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

    int size = 256*vecdim*sizeof(float);
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
}

float get_query_dis(float *query1, float *query2, int vecdim){
    float32x4_t sum_vec = vdupq_n_f32(0);
    for(int k=0;k<vecdim;k+=4){
        float32x4_t vec_1 = vld1q_f32(query1+k);
        float32x4_t vec_2 = vld1q_f32(query2+k);
        float32x4_t mul_vec = vmulq_f32(vec_1,vec_2);
        sum_vec = vaddq_f32(sum_vec,mul_vec);
    }
    float tmp[4];
    vst1q_f32(tmp,sum_vec);
    float dis = tmp[0]+tmp[1]+tmp[2]+tmp[3];
    dis = 1-dis;
    return dis;
}

std::priority_queue<std::pair<float, uint32_t> > ivfpq_search(uint8_t* compressed_base, float *centers, float *centers_dis, float *ivf_centers,std::vector<int> *inverted_kmeans, float* query, size_t base_number, size_t vecdim, size_t k_top) {
    const int K=256;

    int m=6;
    int *query_ivf_labels = new int[m];

    std::priority_queue<std::pair<float, uint32_t> > ivf_q;

    for(int i = 0; i < 256; i++) {
        float dis = get_query_dis(ivf_centers+i*vecdim, query, vecdim);
        if(ivf_q.size() < m) {
            ivf_q.push({dis, i});
        } else {
            if(dis < ivf_q.top().first) {
                ivf_q.push({dis, i});
                ivf_q.pop();
            }
        }
    }

    for(int i=0;i<m;i++){
        query_ivf_labels[i] = ivf_q.top().second;
        ivf_q.pop();
    }

    int *size = new int[m];
    int **candidate_base = new int*[m];
    for(int i=0;i<m;i++){
        size[i] = inverted_kmeans[query_ivf_labels[i]].size();
        candidate_base[i] = inverted_kmeans[query_ivf_labels[i]].data();
    }

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
                float32x4_t mul_vec = vmulq_f32(vec_1,vec_2);
                sum_vec = vaddq_f32(sum_vec,mul_vec);
            }
            float tmp[4];
            vst1q_f32(tmp,sum_vec);
            query_centers_dis[i][j] = tmp[0]+tmp[1]+tmp[2]+tmp[3];
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < m; i++) {
        for(int j=0;j<size[i];j++){
            int candidate_query = candidate_base[i][j];
            float dis = 0;
            for(int k=0;k<4;k++){
                dis+=query_centers_dis[k][compressed_base[candidate_query*4+k]];
            }
            dis = 1-dis;
            if(q.size() < k_top) {
                q.push({dis, candidate_query});
            } else {
                if(dis < q.top().first) {
                    q.push({dis, candidate_query});
                    q.pop();
                }
            }
        }
    }
    return q;
}