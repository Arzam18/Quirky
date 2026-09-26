#ifndef QUIRKY_SRC_UTIL_LARGE_STACK_THREAD_H
#define QUIRKY_SRC_UTIL_LARGE_STACK_THREAD_H

#include <cstddef>
#include <exception>
#include <functional>
#include <pthread.h>
#include <stdexcept>
#include <utility>

// Android's default pthread stack is considerably smaller than the amount
// of stack Quirky's recursive search can use. Quirky also keeps a large
// search context in Searcher. Use an explicitly sized pthread instead of
// relying on std::thread's platform-dependent default.
//
// The stack is virtual address space; pages are committed as they are used.
class LargeStackThread {
  public:
    static constexpr size_t STACK_SIZE = 16ULL * 1024ULL * 1024ULL;

    LargeStackThread() = default;

    template <typename Function>
    explicit LargeStackThread(Function&& function) {
        Start(std::forward<Function>(function));
    }

    LargeStackThread(const LargeStackThread&) = delete;
    LargeStackThread& operator=(const LargeStackThread&) = delete;

    LargeStackThread(LargeStackThread&& other) noexcept
        : thread_(other.thread_), joinable_(other.joinable_) {
        other.joinable_ = false;
    }

    LargeStackThread& operator=(LargeStackThread&& other) noexcept {
        if (this != &other) {
            join();

            thread_ = other.thread_;
            joinable_ = other.joinable_;
            other.joinable_ = false;
        }
        return *this;
    }

    ~LargeStackThread() { join(); }

    bool joinable() const { return joinable_; }

    void join() {
        if (joinable_) {
            pthread_join(thread_, nullptr);
            joinable_ = false;
        }
    }

  private:
    static void* Entry(void* raw) {
        auto function =
            static_cast<std::function<void()>*>(raw);

        try {
            (*function)();
        } catch (...) {
            delete function;
            std::terminate();
        }

        delete function;
        return nullptr;
    }

    template <typename Function>
    void Start(Function&& function) {
        auto* heap_function =
            new std::function<void()>(
                std::forward<Function>(function));

        pthread_attr_t attr;
        int result = pthread_attr_init(&attr);

        if (result != 0) {
            delete heap_function;
            throw std::runtime_error(
                "pthread_attr_init failed");
        }

        result =
            pthread_attr_setstacksize(
                &attr,
                STACK_SIZE);

        if (result != 0) {
            pthread_attr_destroy(&attr);
            delete heap_function;
            throw std::runtime_error(
                "pthread_attr_setstacksize failed");
        }

        result =
            pthread_create(
                &thread_,
                &attr,
                &LargeStackThread::Entry,
                heap_function);

        pthread_attr_destroy(&attr);

        if (result != 0) {
            delete heap_function;
            throw std::runtime_error(
                "pthread_create failed");
        }

        joinable_ = true;
    }

    pthread_t thread_{};
    bool joinable_ = false;
};

#endif  // QUIRKY_SRC_UTIL_LARGE_STACK_THREAD_H
