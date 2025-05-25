#include <pthread.h>
#include <queue>
#include <functional>

class ThreadPool {
private:
    struct Task {
        std::function<void()> func;
    };

    std::vector<pthread_t> workers;
    std::queue<Task> task_queue;
    pthread_mutex_t queue_mutex;
    pthread_cond_t condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false) {
        pthread_mutex_init(&queue_mutex, NULL);
        pthread_cond_init(&condition, NULL);
        
        for(size_t i = 0; i < threads; ++i) {
            pthread_t thread;
            pthread_create(&thread, NULL, [](void* arg) -> void* {
                ThreadPool* pool = static_cast<ThreadPool*>(arg);
                while(true) {
                    pthread_mutex_lock(&pool->queue_mutex);
                    
                    while(pool->task_queue.empty() && !pool->stop) {
                        pthread_cond_wait(&pool->condition, &pool->queue_mutex);
                    }
                    
                    if(pool->stop && pool->task_queue.empty()) {
                        pthread_mutex_unlock(&pool->queue_mutex);
                        return NULL;
                    }
                    
                    Task task = std::move(pool->task_queue.front());
                    pool->task_queue.pop();
                    
                    pthread_mutex_unlock(&pool->queue_mutex);
                    
                    task.func();  // 执行任务
                }
                return NULL;
            }, this);
            workers.push_back(thread);
        }
    }

    template<class F>
    void enqueue(F&& f) {
        pthread_mutex_lock(&queue_mutex);
        task_queue.push(Task{std::forward<F>(f)});
        pthread_cond_signal(&condition);
        pthread_mutex_unlock(&queue_mutex);
    }

    ~ThreadPool() {
        pthread_mutex_lock(&queue_mutex);
        stop = true;
        pthread_cond_broadcast(&condition);
        pthread_mutex_unlock(&queue_mutex);
        
        for(pthread_t &worker : workers) {
            pthread_join(worker, NULL);
        }
        
        pthread_mutex_destroy(&queue_mutex);
        pthread_cond_destroy(&condition);
    }
};