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
#include "flat_scan.h"
#include "kmeans_build.h"
#include "sq.h"
#include "pq.h"
#include "ivfpq.h"
#include "rerank.h"
// 可以自行添加需要的头文件

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


void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
    delete appr_alg;
}

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

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 只测试前2000条查询
    test_number = 1200;

    const size_t k = 10;
    const size_t rerank_k = 100;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // build_index(base, base_number, vecdim);

    //构建sq_compressed_base
    // uint8_t *compressed_base = new uint8_t[base_number*vecdim];
    // sq_compress_base(base,compressed_base, base_number, vecdim);
    // sq_save_compressed_base(compressed_base, base_number, vecdim);
    
    //读取sq_compressed_base
    // uint8_t *compressed_base = new uint8_t[base_number*vecdim];
    // sq_read_compressed_base(compressed_base,base_number,vecdim);

    // 构建pq_kmeans
    // pq_compress_base(base, base_number, vecdim);

    //读取pq_kmeans
    // float *centers = new float[256*vecdim];
    // uint8_t *compressed_base = new uint8_t[base_number*4];
    // float *centers_dis = new float[4*256*256];
    // pq_read_kmeans(centers, compressed_base, centers_dis, base_number, vecdim);

    //构建ivf
    // int M=256;
    // float *centers = new float[M*vecdim];
    // uint8_t *labels = new uint8_t[base_number];
    // kmeans(base, labels, centers, base_number, vecdim, M);
    // build_inverted_kmeans(labels, centers, base_number, vecdim, M);

    //构建ivfpq
    // int M=256;
    // std::vector<int>* inverted_kmeans = new std::vector<int>[M];
    // float* ivf_centers = new float[M*vecdim];
    // uint8_t *ivf_base = new uint8_t[base_number];
    // read_inverted_kmeans(inverted_kmeans, ivf_centers, ivf_base, base_number, vecdim, M);
    // ivfpq_compress_base(base, ivf_base, ivf_centers, base_number, vecdim);

    // 读取ivfpq
    int M=256;
    std::vector<int>* inverted_kmeans = new std::vector<int>[M];
    float* ivf_centers = new float[M*vecdim];
    uint8_t *ivf_base = new uint8_t[base_number];
    read_inverted_kmeans(inverted_kmeans, ivf_centers, ivf_base, base_number, vecdim, M);
    float *ivfpq_centers = new float[256*vecdim];
    uint8_t *ivfpq_compressed_base = new uint8_t[base_number*4];
    ivfpq_read_kmeans(ivfpq_centers, ivfpq_compressed_base, base_number, vecdim);

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timespec val;
        clock_gettime(CLOCK_MONOTONIC, &val);

        //暴力搜索
        // auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

        //sq搜索
        // uint8_t *compressed_query = new uint8_t[vecdim];
        // sq_compress_base(test_query+i*vecdim, compressed_query, 1, vecdim);
        // auto raw_res = sq_search(compressed_base, compressed_query, base_number, vecdim, k);

        //pq搜索
        // auto raw_res = pq_search(compressed_base, centers, centers_dis, test_query+i*vecdim, base_number, vecdim, rerank_k);

        //ivfpq搜索
        int m=8;
        auto raw_res = ivfpq_search(ivfpq_compressed_base, ivfpq_centers, ivf_centers, inverted_kmeans, test_query+i*vecdim, base_number, vecdim, rerank_k, m);

        auto res = rerank(base, raw_res, test_query+i*vecdim, vecdim, k);
        

        struct timespec newVal;
        clock_gettime(CLOCK_MONOTONIC, &newVal);
        int64_t diff = (newVal.tv_sec - val.tv_sec) * 1000000LL 
             + (newVal.tv_nsec - val.tv_nsec) / 1000;

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";

    return 0;
}
