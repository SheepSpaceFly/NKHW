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
