#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include <iostream>
#include <fstream>
#include <cstddef>
#include <queue>
#include <algorithm>
#include <pthread.h>
#include <mpi.h>
#include <arm_neon.h>

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

int64_t time_diff_us(const struct timeval& start, const struct timeval& end) {
    int64_t diff_sec = end.tv_sec - start.tv_sec;
    int64_t diff_usec = end.tv_usec - start.tv_usec;
    // 处理微秒借位
    if (diff_usec < 0) {
        diff_sec -= 1;
        diff_usec += 1000000;
    }
    return diff_sec * 1000000 + diff_usec;
}


float get_query_dis_rerank(float *query1, float *query2, int vecdim){
    float sum = 0.0f;
    for(int k = 0; k < vecdim; k++){
        sum += query1[k] * query2[k];
    }
    return 1 - sum;
}

std::priority_queue<std::pair<float, uint32_t>> rerank(float* base, std::priority_queue<std::pair<float, uint32_t>> results, float* query, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    while (results.size()) {   
        int x = results.top().second;
        float dis = 0; 

        dis = get_query_dis_rerank(base+x*vecdim,query,vecdim);

        if(q.size() < k) {
            q.push({dis, x});
        } else {
            if(dis < q.top().first) {
                q.push({dis, x});
                q.pop();
            }
        }
        results.pop();
    }

    return q;
}

void ivfpq_read_kmeans(float* centers, uint8_t*compressed_base, size_t base_number, size_t vecdim){
    const int K = 256;
    int size = base_number*4;
    std::ifstream in_file_1("./files/ivfpq_kmeans_base.bin",std::ios::binary);
    if (!in_file_1.is_open()) {
        std::cout<<"无法打开ivfpq_kmeans_base文件！"<<std::endl;
        return;
    }
    if(!in_file_1.read(reinterpret_cast<char*>(compressed_base),size)){
        std::cout<<"读入ivfpq_kmeans_base.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入ivfpq_kmeans_base.bin！"<<std::endl;


    size = K*vecdim*sizeof(float);
    std::ifstream in_file_2("./files/ivfpq_kmeans_centers.bin",std::ios::binary);
    if (!in_file_2.is_open()) {
        std::cout<<"无法打开ivfpq_kmeans_centers文件！"<<std::endl;
        return;
    }
    if(!in_file_2.read(reinterpret_cast<char*>(centers),size)){
        std::cout<<"读入ivfpq_kmeans_centers.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入ivfpq_kmeans_centers.bin！"<<std::endl;
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


int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);
    
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    float* test_query = nullptr;
    int* test_gt = nullptr;
    float* base = nullptr;
    
    if (rank == 0) {
        std::string data_path = "/anndata/"; 
        test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
        test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
        
        test_number = 1200;
    }
    
    MPI_Bcast(&test_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_gt_d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    int M = 256;
    const size_t k = 10;
    const size_t rerank_k = 300;
    const int thread_number = 2;
    int m = 24;
    
    std::vector<int>* inverted_kmeans = new std::vector<int>[M];
    float* ivf_centers = new float[M * vecdim];
    uint8_t* ivf_base = new uint8_t[base_number];
    float* ivfpq_centers = new float[256 * vecdim];
    uint8_t* ivfpq_compressed_base = new uint8_t[base_number * 4];
    
    if (rank == 0) {
        read_inverted_kmeans(inverted_kmeans, ivf_centers, ivf_base, base_number, vecdim, M);
        ivfpq_read_kmeans(ivfpq_centers, ivfpq_compressed_base, base_number, vecdim);
    }
    
    int* lens = new int[M];
    if (rank == 0) {
        for (int i = 0; i < M; i++) {
            lens[i] = inverted_kmeans[i].size();
        }
    }
    MPI_Bcast(lens, M, MPI_INT, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < M; i++) {
        if (rank != 0) {
            inverted_kmeans[i].resize(lens[i]);
        }
        MPI_Bcast(inverted_kmeans[i].data(), lens[i], MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Bcast(ivf_centers, M * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ivf_base, base_number, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(ivfpq_centers, 256 * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ivfpq_compressed_base, base_number * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    delete[] lens;
    
    std::vector<SearchResult> results;
    if (rank == 0) {
        results.resize(test_number);
    }

    struct timespec val;
    
    for (int i = 0; i < test_number; ++i) {
        float* query = new float[vecdim];
        if (rank == 0) {
            memcpy(query, test_query + i * vecdim, vecdim * sizeof(float));
        }
        MPI_Bcast(query, vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        int64_t diff = 0;
        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &val);
        }
        
        // 在rank0找到最近的m个簇
        int* query_ivf_labels = new int[m];
        if (rank == 0) {
            std::priority_queue<std::pair<float, uint32_t> > ivf_q;
            
            for (int i_label = 0; i_label < M; i_label++) {
                float dis = get_query_dis(ivf_centers + i_label * vecdim, query, vecdim);
                if (ivf_q.size() < static_cast<size_t>(m)) {
                    ivf_q.push(std::make_pair(dis, i_label));
                } else if (dis < ivf_q.top().first) {
                    ivf_q.pop();
                    ivf_q.push(std::make_pair(dis, i_label));
                }
            }
            
            for (int j = m - 1; j >= 0; j--) {
                query_ivf_labels[j] = ivf_q.top().second;
                ivf_q.pop();
            }
        }
        
        MPI_Bcast(query_ivf_labels, m, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 每个进程处理部分簇
        int clusters_per_proc = (m + world_size - 1) / world_size;
        int start_idx = std::min(rank * clusters_per_proc, m);
        int end_idx = std::min((rank + 1) * clusters_per_proc, m);
        int local_cluster_count = end_idx - start_idx;
        

        size_t local_candidate_count = 0;
        for (int j = start_idx; j < end_idx; j++) {
            int label = query_ivf_labels[j];
            local_candidate_count += inverted_kmeans[label].size();
        }
        
        int* local_candidate_queries = new int[local_candidate_count];
        int* local_residual_i = new int[local_candidate_count];
        float** local_residual = new float*[local_cluster_count];
        
        size_t count_idx = 0;
        for (int j = 0; j < local_cluster_count; j++) {
            int global_idx = start_idx + j;
            int label = query_ivf_labels[global_idx];
            
            local_residual[j] = new float[vecdim];
            float* center_ptr = ivf_centers + label * vecdim;
            for (int d = 0; d < vecdim; d++) {
                local_residual[j][d] = query[d] - center_ptr[d];
            }
            
            for (size_t k = 0; k < inverted_kmeans[label].size(); k++) {
                local_candidate_queries[count_idx] = inverted_kmeans[label][k];
                local_residual_i[count_idx] = j;
                count_idx++;
            }
        }
        
        // 本地距离计算
        std::priority_queue<std::pair<float, uint32_t>> local_main_q;
        for (size_t idx = 0; idx < local_candidate_count; idx++) {
            int candidate_query = local_candidate_queries[idx];
            float dis = 0;
            
            for (int k_sub = 0; k_sub < 4; k_sub++) {
                int center_index = ivfpq_compressed_base[candidate_query * 4 + k_sub];
                float* sub_center = ivfpq_centers + k_sub * 256 * (vecdim / 4) + center_index * (vecdim / 4);
                float* sub_residual = local_residual[local_residual_i[idx]] + k_sub * (vecdim / 4);
                dis += get_query_dis(sub_center, sub_residual, vecdim / 4);
            }
            
            if (local_main_q.size() < rerank_k) {
                local_main_q.push(std::make_pair(dis, candidate_query));
            } else if (dis < local_main_q.top().first) {
                local_main_q.pop();
                local_main_q.push(std::make_pair(dis, candidate_query));
            }
        }

        //OpenMP多线程版本
        // std::priority_queue<std::pair<float, uint32_t> > local_main_q;
        
        // #pragma omp parallel num_threads(thread_number)
        // {
        //     std::priority_queue<std::pair<float, uint32_t> > local_q;
            
        //     #pragma omp for schedule(dynamic)
        //     for (size_t idx = 0; idx < local_candidate_count; idx++) {
        //         int candidate_query = local_candidate_queries[idx];
        //         float dis = 0;
                
        //         // 计算距离：遍历4个子空间
        //         for (int k_sub = 0; k_sub < 4; k_sub++) {
        //             int center_index = ivfpq_compressed_base[candidate_query * 4 + k_sub];
        //             float* sub_center = ivfpq_centers + k_sub * 256 * (vecdim / 4) + center_index * (vecdim / 4);
        //             float* sub_residual = local_residual[local_residual_i[idx]] + k_sub * (vecdim / 4);
        //             dis += get_query_dis(sub_center, sub_residual, vecdim / 4);
        //         }
                
        //         if (local_q.size() < rerank_k) {
        //             local_q.push(std::make_pair(dis, candidate_query));
        //         } else if (dis < local_q.top().first) {
        //             local_q.pop();
        //             local_q.push(std::make_pair(dis, candidate_query));
        //         }
        //     }
            
        //     #pragma omp critical
        //     {
        //         while (!local_q.empty()) {
        //             auto elem = local_q.top();
        //             if (local_main_q.size() < rerank_k) {
        //                 local_main_q.push(elem);
        //             } else if (elem.first < local_main_q.top().first) {
        //                 local_main_q.pop();
        //                 local_main_q.push(elem);
        //             }
        //             local_q.pop();
        //         }
        //     }
        // }
        
        int local_result_count = local_main_q.size();
        float* local_dists = new float[local_result_count];
        uint32_t* local_indices = new uint32_t[local_result_count];
        
        size_t pos = 0;
        while (!local_main_q.empty()) {
            auto elem = local_main_q.top();
            local_dists[pos] = elem.first;
            local_indices[pos] = elem.second;
            local_main_q.pop();
            pos++;
        }
        
        // 在rank0汇总并处理
        if (rank == 0) {
            std::priority_queue<std::pair<float, uint32_t> > global_main_q;
            
            for (int j = 0; j < local_result_count; j++) {
                global_main_q.push(std::make_pair(local_dists[j], local_indices[j]));
            }
            
            // 接收其他进程的结果
            for (int r = 1; r < world_size; r++) {
                int recv_count;
                MPI_Recv(&recv_count, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                float* recv_dists = new float[recv_count];
                uint32_t* recv_indices = new uint32_t[recv_count];
                
                MPI_Recv(recv_dists, recv_count, MPI_FLOAT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_indices, recv_count, MPI_UNSIGNED, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (int j = 0; j < recv_count; j++) {
                    if (global_main_q.size() < rerank_k) {
                        global_main_q.push(std::make_pair(recv_dists[j], recv_indices[j]));
                    } else if (recv_dists[j] < global_main_q.top().first) {
                        global_main_q.pop();
                        global_main_q.push(std::make_pair(recv_dists[j], recv_indices[j]));
                    }
                }
                
                delete[] recv_dists;
                delete[] recv_indices;
            }

            auto res = rerank(base, global_main_q, test_query + i * vecdim, vecdim, k);
            
            struct timespec newVal;
            clock_gettime(CLOCK_MONOTONIC, &newVal);
            diff = (newVal.tv_sec - val.tv_sec) * 1000000LL + (newVal.tv_nsec - val.tv_nsec) / 1000;
            
            std::set<uint32_t> gtset;
            for (int j = 0; j < k; j++) {
                int t = test_gt[j + i * test_gt_d];
                gtset.insert(t);
            }
            
            size_t acc = 0;
            while (!res.empty()) {
                int x = res.top().second;
                if (gtset.find(x) != gtset.end()) {
                    ++acc;
                }
                res.pop();
            }
            float recall = (float)acc / k;
            
            results[i] = {recall, diff};
        }
        else{
            MPI_Send(&local_result_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(local_dists, local_result_count, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(local_indices, local_result_count, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD);
        }
        
        delete[] query;
        delete[] query_ivf_labels;
        delete[] local_candidate_queries;
        delete[] local_residual_i;
        for (int j = 0; j < local_cluster_count; j++) {
            delete[] local_residual[j];
        }
        delete[] local_residual;
        delete[] local_dists;
        delete[] local_indices;
    }
    
    if (rank == 0) {
        float avg_recall = 0, avg_latency = 0;
        for (int i = 0; i < test_number; i++) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }
        
        std::cout << "average recall: " << avg_recall / test_number << "\n";
        std::cout << "average latency (us): " << avg_latency / test_number << "\n";
        
        delete[] test_query;
        delete[] test_gt;
        delete[] base;
    }
    
    delete[] inverted_kmeans;
    delete[] ivf_centers;
    delete[] ivf_base;
    delete[] ivfpq_centers;
    delete[] ivfpq_compressed_base;
    
    MPI_Finalize();
    return 0;
}
