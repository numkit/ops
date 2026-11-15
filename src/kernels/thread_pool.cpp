#include <keel/core.h>
#include <numkit/ops/thread_pool.hpp>

#include <algorithm>
#include <utility>
#include <hwy/cache_control.h>

namespace numkit::detail {

inline void spin_pause() noexcept {
    hwy::Pause();
}

// True while the current thread is executing a pool task body. A nested
// run() (a parallel body that itself calls parallel_for) is then run inline
// rather than re-dispatched, which would deadlock.
namespace {
thread_local bool t_inPool = false;
}

ThreadPool &ThreadPool::global()
{
    // hardware_concurrency() can return 0 when the runtime has no idea
    // (rare on desktop, occasionally on WSL/containers); fall back to 1
    // so the pool is well-formed (and `run()` then takes the sequential
    // shortcut anyway).
    static const int n = std::max(1u, std::thread::hardware_concurrency());
    static ThreadPool instance(n);
    return instance;
}

ThreadPool::ThreadPool(int n_workers)
{
    if (n_workers <= 1)
        return; // workers_ stays empty; run() degrades to fn(0, n)
    workers_.reserve(static_cast<keel::usize>(n_workers));
    for (int i = 0; i < n_workers; ++i)
        workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mu_);
        shutdown_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_release);
    }
    cv_start_.notify_all();
    for (auto &t : workers_)
        if (t.joinable())
            t.join();
}

void ThreadPool::run(keel::usize n, std::function<void(keel::usize, keel::usize)> fn,
                     int max_workers)
{
    if (n == 0)
        return;

    // Nested dispatch (a worker's task body itself called parallel_for): run
    // inline. Re-dispatching to the same pool would deadlock — this thread
    // would block on sibling chunks that cannot make progress.
    if (t_inPool) {
        fn(0, n);
        return;
    }

    int k = workers();
    if (max_workers > 0 && max_workers < k)
        k = max_workers;
    if (k <= 1) {
        // Pool unused or capped to 1 — run on the caller's thread.
        fn(0, n);
        return;
    }

    // Serialize concurrent submitters (the pool's task_/epoch_ slots are
    // single-task). Held across publish+wait; workers use mu_, not this, and a
    // nested run() already returned above, so this is never contended by a
    // worker.
    std::lock_guard<std::mutex> submitLock(submit_mu_);

    {
        std::unique_lock<std::mutex> lock(mu_);
        task_           = std::move(fn);
        task_n_         = n;
        task_remaining_.store(k, std::memory_order_release);
        active_         = k;
        epoch_.fetch_add(1, std::memory_order_release);
    }
    cv_start_.notify_all();

    {
        for (int spin = 0; spin < 2000; ++spin) {
            if (task_remaining_.load(std::memory_order_acquire) == 0) break;
            spin_pause();
        }
        std::function<void(keel::usize, keel::usize)> cleanup;
        {
            std::unique_lock<std::mutex> lock(mu_);
            if (task_remaining_.load(std::memory_order_acquire) != 0) {
                cv_done_.wait(lock, [this] {
                    return task_remaining_.load(std::memory_order_acquire) == 0;
                });
            }
            // Reset active_ and clear task_ under mu_ so any worker waking up
            // from an earlier broadcast sees a consistent idle state instead
            // of racing on task_ / active_ / task_n_.
            active_ = 0;
            cleanup = std::move(task_);
            task_ = nullptr;
        }
        (void)cleanup;
    }
}

void ThreadPool::worker_loop(int id)
{
    keel::u64 seen_epoch = 0;
    while (true) {
        std::function<void(keel::usize, keel::usize)> fn;
        keel::usize n;
        int         k;

        {
            for (int spin = 0; spin < 2000; ++spin) {
                if (shutdown_.load(std::memory_order_acquire)
                    || epoch_.load(std::memory_order_acquire) != seen_epoch)
                    break;
                spin_pause();
            }
            std::unique_lock<std::mutex> lock(mu_);
            cv_start_.wait(lock, [&] {
                return shutdown_.load(std::memory_order_acquire)
                    || epoch_.load(std::memory_order_acquire) != seen_epoch;
            });
            if (shutdown_.load(std::memory_order_acquire))
                return;
            seen_epoch = epoch_.load(std::memory_order_relaxed);
            // Workers beyond the active cap (or waking after the epoch already
            // completed and reset active_ = 0) skip this task cleanly under mu_.
            if (id >= active_)
                continue;
            fn         = task_;
            n          = task_n_;
            k          = active_;
        }

        const keel::usize chunk = (n + static_cast<keel::usize>(k) - 1)
                                  / static_cast<keel::usize>(k);
        const keel::usize start = static_cast<keel::usize>(id) * chunk;
        const keel::usize end   = std::min(start + chunk, n);
        if (start < end) {
            t_inPool = true;     // mark so a nested run() on this thread inlines
            fn(start, end);
            t_inPool = false;
        }

        {
            if (task_remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mu_);
                cv_done_.notify_one();
            }
        }
    }
}

} // namespace numkit::detail
