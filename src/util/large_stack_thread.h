#ifndef QUIRKY_SRC_UTIL_LARGE_STACK_THREAD_H
#define QUIRKY_SRC_UTIL_LARGE_STACK_THREAD_H

#include <cstddef>
#include <exception>
#include <functional>
#include <stdexcept>
#include <utility>

#if defined(__ANDROID__)
#include <pthread.h>
#else
#include <thread>
#endif

// Android's default pthread stack is too small for Quirky's Searcher
// construction and recursive search frames. Reserve a larger stack for both
// the launcher thread and the actual search thread. The OS commits pages as
// they are touched; this does not allocate 32 MiB of physical RAM per thread.
class LargeStackThread {
  public:
    static constexpr size_t STACK_SIZE = 32ULL * 1024ULL * 1024ULL;

    LargeStackThread() = default;

    template <typename Function>
    explicit LargeStackThread(Function&& function) {
        Start(std::forward<Function>(function));
    }

    LargeStackThread(const LargeStackThread&) = delete;
    LargeStackThread& operator=(const LargeStackThread&) = delete;

    LargeStackThread(LargeStackThread&& other) noexcept
#if defined(__ANDROID__)
        : thread_(other.thread_), joinable_(other.joinable_) {
#else
        : thread_(std::move(other.thread_)), joinable_(other.joinable_) {
#endif
        other.joinable_ = false;
    }

    LargeStackThread& operator=(LargeStackThread&& other) noexcept {
        if (this != &other) {
            join();
#if defined(__ANDROID__)
            thread_ = other.thread_;
#else
            thread_ = std::move(other.thread_);
#endif
            joinable_ = other.joinable_;
            other.joinable_ = false;
        }
        return *this;
    }

    ~LargeStackThread() { join(); }

    bool joinable() const noexcept { return joinable_; }

    void join() {
        if (!joinable_) {
            return;
        }
#if defined(__ANDROID__)
        const int result = pthread_join(thread_, nullptr);
        joinable_ = false;
        if (result != 0) {
            std::terminate();
        }
#else
        if (thread_.joinable()) {
            thread_.join();
        }
        joinable_ = false;
#endif
    }

  private:
#if defined(__ANDROID__)
    static void* Entry(void* raw) {
        auto function = static_cast<std::function<void()>*>(raw);
        try {
            (*function)();
        } catch (...) {
            delete function;
            std::terminate();
        }
        delete function;
        return nullptr;
    }
#endif

    template <typename Function>
    void Start(Function&& function) {
#if defined(__ANDROID__)
        auto* heap_function =
            new std::function<void()>(std::forward<Function>(function));

        pthread_attr_t attr;
        int result = pthread_attr_init(&attr);
        if (result != 0) {
            delete heap_function;
            throw std::runtime_error("pthread_attr_init failed");
        }

        result = pthread_attr_setstacksize(&attr, STACK_SIZE);
        if (result != 0) {
            pthread_attr_destroy(&attr);
            delete heap_function;
            throw std::runtime_error("pthread_attr_setstacksize failed");
        }

        result = pthread_create(&thread_, &attr, &LargeStackThread::Entry,
                                heap_function);
        pthread_attr_destroy(&attr);
        if (result != 0) {
            delete heap_function;
            throw std::runtime_error("pthread_create failed");
        }
        joinable_ = true;
#else
        thread_ = std::thread(std::forward<Function>(function));
        joinable_ = true;
#endif
    }

#if defined(__ANDROID__)
    pthread_t thread_{};
#else
    std::thread thread_;
#endif
    bool joinable_ = false;
};

#endif  // QUIRKY_SRC_UTIL_LARGE_STACK_THREAD_H
