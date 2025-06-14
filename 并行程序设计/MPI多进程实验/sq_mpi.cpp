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
#include <arm_neon.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <queue>
#include <algorithm>
#include <pthread.h>
#include <mpi.h>

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


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    float *test_query = nullptr;
    int *test_gt = nullptr;
    float *base = nullptr;
    uint8_t *global_compressed_base = nullptr;

    if (rank == 0) {
        std::string data_path = "/anndata/";
        test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
        test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

        test_number = 1200;
    }

    MPI_Bcast(&test_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);  
    MPI_Bcast(&vecdim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    int chunk_size = base_number / world_size;
    int remainder = base_number % world_size;
    int start = rank * chunk_size + (rank < remainder ? rank : remainder);
    int end = start + chunk_size + (rank < remainder ? 1 : 0);
    size_t local_base_number = end - start;

    uint8_t *local_compressed_base = new uint8_t[local_base_number * vecdim];
    {
        std::ifstream in_file("./files/sq_compressed_base.bin", std::ios::binary);
        if (!in_file.is_open()) {
            std::cerr << "无法打开sq_compressed_base.bin文件！" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t offset = start * vecdim * sizeof(uint8_t);
        in_file.seekg(offset, std::ios::beg);
        in_file.read(reinterpret_cast<char*>(local_compressed_base), local_base_number * vecdim * sizeof(uint8_t));
        in_file.close();
    }

    const size_t k = 10;
    const size_t rerank_k = 200;
    uint8_t *compressed_query = new uint8_t[vecdim];
    float avg_recall = 0, avg_latency = 0;
    int thread_number = 4;

    for (int i = 0; i < test_number; i++) {
        struct timespec val, newVal;
        if (rank == 0) {
            sq_compress_base(test_query + i * vecdim, compressed_query, 1, vecdim);
            clock_gettime(CLOCK_MONOTONIC, &val);
        }

        MPI_Bcast(compressed_query, vecdim, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        std::priority_queue<std::pair<float, uint32_t>> local_q;
        for (size_t idx = 0; idx < local_base_number; ++idx) {
            float dis = sq_l2_distance(
                compressed_query, 
                local_compressed_base + idx * vecdim, 
                vecdim
            );
            
            if (local_q.size() < rerank_k) {
                local_q.push({dis, static_cast<uint32_t>(start + idx)});
            } else if (dis < local_q.top().first) {
                local_q.pop();
                local_q.push({dis, static_cast<uint32_t>(start + idx)});
            }
        }

        //OpenMP并行版本
        // std::priority_queue<std::pair<float, uint32_t>> local_q;
        // #pragma omp parallel num_threads(thread_number)
        // {
        //     std::priority_queue<std::pair<float, uint32_t>> tmp_q;
            
        //     #pragma omp for schedule(static)
        //     for (size_t idx = 0; idx < local_base_number; ++idx) {
        //         float dis = sq_l2_distance(
        //             compressed_query, 
        //             local_compressed_base + idx * vecdim, 
        //             vecdim
        //         );
                
        //         if (tmp_q.size() < rerank_k) {
        //             tmp_q.push({dis, static_cast<uint32_t>(start + idx)});
        //         } else if (dis < tmp_q.top().first) {
        //             tmp_q.pop();
        //             tmp_q.push({dis, static_cast<uint32_t>(start + idx)});
        //         }
        //     }

        //     #pragma omp critical
        //     {
        //         while (!tmp_q.empty()) {
        //             const auto& item = tmp_q.top();
        //             // 处理全局队列
        //             if (local_q.size() < rerank_k) {
        //                 local_q.push(item);
        //             } else if (item.first < local_q.top().first) {
        //                 local_q.pop();
        //                 local_q.push(item);
        //             }
        //             tmp_q.pop();
        //         }
        //     }
        // }

        if (rank == 0) {
            std::vector<std::pair<float, uint32_t>> all_candidates;
            while (!local_q.empty()) {
                all_candidates.push_back(local_q.top());
                local_q.pop();
            }

            for (int r = 1; r < world_size; ++r) {
                int count;
                MPI_Recv(&count, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                std::vector<float> recv_dists(count);
                std::vector<uint32_t> recv_indices(count);
                MPI_Recv(recv_dists.data(), count, MPI_FLOAT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_indices.data(), count, MPI_UNSIGNED, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (int j = 0; j < count; ++j) {
                    all_candidates.push_back({recv_dists[j], recv_indices[j]});
                }
            }

            std::priority_queue<std::pair<float, uint32_t>> global_q;
            for (const auto& cand : all_candidates) {
                if (global_q.size() < rerank_k) {
                    global_q.push(cand);
                } else if (cand.first < global_q.top().first) {
                    global_q.pop();
                    global_q.push(cand);
                }
            }

            auto res = rerank(base, global_q, test_query + i * vecdim, vecdim, k);

            clock_gettime(CLOCK_MONOTONIC, &newVal);
            int64_t diff = (newVal.tv_sec - val.tv_sec) * 1000000LL + (newVal.tv_nsec - val.tv_nsec) / 1000;

            std::set<uint32_t> gtset;
            for (int j = 0; j < k; ++j) {
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

            avg_recall += recall;
            avg_latency += diff;
        } else {
            int count = local_q.size();
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            std::vector<float> dists;
            std::vector<uint32_t> indices;
            while (!local_q.empty()) {
                dists.push_back(local_q.top().first);
                indices.push_back(local_q.top().second);
                local_q.pop();
            }
            
            MPI_Send(dists.data(), count, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(indices.data(), count, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        std::cout << "average recall: " << avg_recall / test_number << "\n";
        std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    }

    // 清理资源
    delete[] compressed_query;
    delete[] local_compressed_base;
    if (rank == 0) {
        delete[] test_query;
        delete[] test_gt;
        delete[] base;
    }

    MPI_Finalize();
    return 0;
}
