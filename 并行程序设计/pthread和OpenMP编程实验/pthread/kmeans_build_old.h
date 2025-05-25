#include <arm_neon.h>
#include <iostream>
#include <fstream>
#include <cstddef>
#include <queue>

void kmeans(float* data, uint8_t* labels, float* centers, int num_points, int dims, int num_clusters, int max_iter = 10){
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    for (int k = 0; k < num_clusters; ++k) {
        int idx = std::rand() % num_points;
        float norm = 0.0f;
        
        for (int j = 0; j < dims; ++j) {
            centers[k * dims + j] = data[idx * dims + j];
            norm += centers[k * dims + j] * centers[k * dims + j];
        }
        norm = std::sqrt(norm);

        if (norm > 1e-6) {
            for (int j = 0; j < dims; ++j) {
                centers[k * dims + j] /= norm;
            }
        }
    }

    std::vector<float> old_centers(num_clusters * dims);

    for (int iter = 0; iter < max_iter; ++iter) {
        std::cout<<iter<<std::endl;
        for (int i = 0; i < num_points; ++i) {
            float max_inner_product = -std::numeric_limits<float>::max();
            int best_cluster = 0;

            for (int k = 0; k < num_clusters; ++k) {
                float inner_product = 0.0f;
                const float* center = &centers[k * dims];
                const float* vec = &data[i * dims];

                float32x4_t sum_vec = vdupq_n_f32(0);
                int j = 0;
                for (; j <= dims - 4; j += 4) {
                    float32x4_t v_data = vld1q_f32(vec + j);
                    float32x4_t v_cent = vld1q_f32(center + j);
                    sum_vec = vmlaq_f32(sum_vec, v_data, v_cent);
                }

                float sum = vaddvq_f32(sum_vec);
                for (; j < dims; ++j) {
                    sum += vec[j] * center[j];
                }

                if (sum > max_inner_product) {
                    max_inner_product = sum;
                    best_cluster = k;
                }
            }
            labels[i] = static_cast<uint8_t>(best_cluster);
        }

        std::copy(centers, centers + num_clusters * dims, old_centers.begin());
        std::vector<int> counts(num_clusters, 0);
        std::fill(centers, centers + num_clusters * dims, 0.0f);

        for (int i = 0; i < num_points; ++i) {
            int cluster = labels[i];
            const float* vec = &data[i * dims];
            float* center = &centers[cluster * dims];
            
            for (int j = 0; j < dims; ++j) {
                center[j] += vec[j];
            }
            counts[cluster]++;
        }

        for (int k = 0; k < num_clusters; ++k) {
            if (counts[k] > 0) {
                float* center = &centers[k * dims];
                float norm = 0.0f;

                for (int j = 0; j < dims; ++j) {
                    center[j] /= counts[k];
                }

                for (int j = 0; j < dims; ++j) {
                    norm += center[j] * center[j];
                }
                norm = std::sqrt(norm);

                if (norm > 1e-6) {
                    for (int j = 0; j < dims; ++j) {
                        center[j] /= norm;
                    }
                }
            } else {
                int idx = std::rand() % num_points;
                for (int j = 0; j < dims; ++j) {
                    centers[k * dims + j] = data[idx * dims + j];
                }
            }
        }

        bool converged = true;
        for (int k = 0; k < num_clusters * dims; ++k) {
            if (std::abs(centers[k] - old_centers[k]) > 1e-5) {
                converged = false;
                break;
            }
        }
        if (converged) break;
    }
}

void kmeans_compress_base(float *base, size_t base_number, size_t vecdim){
    const int K = 256;//聚类中心数
    uint8_t *compressed_base = new uint8_t[base_number*4];
    float *centers = new float[vecdim*K];
    
    for(int i=0;i<4;i++){
        float *tmp_base = new float[base_number*vecdim/4];
        for(int j=0;j<base_number;j++){
            for(int k=0;k<vecdim/4;k++){
                tmp_base[j*vecdim/4+k] = base[i*vecdim/4+j*vecdim+k];
            }
        }

        uint8_t *tmp_labels = new uint8_t[base_number];
        float* tmp_centers = new float[vecdim/4*K];

        kmeans(tmp_base,tmp_labels,tmp_centers,base_number,vecdim/4,K);

        for(int j=0;j<base_number;j++){
            compressed_base[j*4+i] = tmp_labels[j];
        }

        for(int j=0;j<K;j++){
            for(int k=0;k<vecdim/4;k++){
                centers[(i*K + j)*vecdim/4 + k] = tmp_centers[j*vecdim/4 + k];
            }
        }
        delete[] tmp_base;
        delete[] tmp_labels;
        delete[] tmp_centers;
    }

    float *centers_dis = new float[4*256*256];

    for(int i=0;i<4;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<256;k++){
                float *center_1 = centers+(i*K + j)*vecdim/4;
                float *center_2 = centers+(i*K + k)*vecdim/4;
                float sum = 0;
                for(int l=0;l<vecdim/4;l++){
                    sum+=center_1[l]*center_2[l];
                }
                centers_dis[i*256*256+j*256+k] = sum;
            }
        }
    }

    int size = base_number*4;
    std::ofstream out_file_1("./files/pq_kmeans_base.bin",std::ios::binary);
    if(!out_file_1){
        std::cout<<"无法打开pq_kmeans_base.bin文件！"<<std::endl;
        return;
    }

    out_file_1.write(reinterpret_cast<const char*>(compressed_base), size);

    if(!out_file_1.good()){
        std::cout<<"写入pq_kmeans_base.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入pq_kmeans_base.bin！"<<std::endl;

    size = K*vecdim*sizeof(float);
    std::ofstream out_file_2("./files/pq_kmeans_centers.bin",std::ios::binary);
    if(!out_file_2){
        std::cout<<"无法打开pq_kmeans_base.bin文件！"<<std::endl;
        return;
    }

    out_file_2.write(reinterpret_cast<const char*>(centers), size);

    if(!out_file_2.good()){
        std::cout<<"写入pq_kmeans_centers.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入pq_kmeans_centers.bin！"<<std::endl;

    size = 4*256*256*sizeof(float);
    std::ofstream out_file_3("./files/pq_kmeans_centers_distance.bin",std::ios::binary);
    if(!out_file_3){
        std::cout<<"无法打开pq_kmeans_centers_distance.bin文件！"<<std::endl;
        return;
    }

    out_file_3.write(reinterpret_cast<const char*>(centers_dis), size);

    if(!out_file_3.good()){
        std::cout<<"写入pq_kmeans_centers_distance.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功写入pq_kmeans_centers_distance.bin！"<<std::endl;

    delete[] compressed_base;
    delete[] centers;

}

void read_kmeans(float* centers, uint8_t*compressed_base, float *centers_dis, size_t base_number, size_t vecdim){
    const int K = 256;
    int size = base_number*4;
    std::ifstream in_file_1("./files/pq_kmeans_base.bin",std::ios::binary);
    if (!in_file_1.is_open()) {
        std::cout<<"无法打开pq_kmeans_base文件！"<<std::endl;
        return;
    }
    if(!in_file_1.read(reinterpret_cast<char*>(compressed_base),size)){
        std::cout<<"读入pq_kmeans_base.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入pq_kmeans_base.bin！"<<std::endl;


    size = K*vecdim*sizeof(float);
    std::ifstream in_file_2("./files/pq_kmeans_centers.bin",std::ios::binary);
    if (!in_file_2.is_open()) {
        std::cout<<"无法打开pq_kmeans_centers文件！"<<std::endl;
        return;
    }
    if(!in_file_2.read(reinterpret_cast<char*>(centers),size)){
        std::cout<<"读入pq_kmeans_centers.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入pq_kmeans_centers.bin！"<<std::endl;

    size = 256*256*4*sizeof(float);
    std::ifstream in_file_3("./files/pq_kmeans_centers_distance.bin",std::ios::binary);
    if (!in_file_3.is_open()) {
        std::cout<<"无法打开pq_kmeans_centers_distance文件！"<<std::endl;
        return;
    }
    if(!in_file_3.read(reinterpret_cast<char*>(centers_dis),size)){
        std::cout<<"读入pq_kmeans_centers_distance.bin出错！"<<std::endl;
        return;
    }
    std::cout<<"成功读入pq_kmeans_centers_distance.bin！"<<std::endl;
}