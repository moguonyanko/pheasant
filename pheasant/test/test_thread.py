import logging
import threading
import time

# 共有リソース
counter = 0
lock = threading.Lock()

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# カウントする回数（重めの処理にするために大きめに設定）
ITERATIONS = 5_000_000


def increment_task_efficient():
    global counter
    local_count = 0  # スレッド内だけでカウントする
    for _ in range(ITERATIONS):
        local_count += 1

    # 最後に1回だけロックして共有変数に加算
    # ロックの回数を減らすことで高速化を図る。
    with lock:
        counter += local_count


def run_experiment(thread_count, task_func):
    global counter
    counter = 0
    threads = []
    start_time = time.perf_counter()

    for _ in range(thread_count):
        t = threading.Thread(target=task_func)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.perf_counter()
    logger.debug(f"スレッド数: {thread_count} | 実行時間: {end_time - start_time:.4f}秒")


def test_threading_with_gil():
    logger.debug("--- 1スレッドで実行 ---")
    run_experiment(1, increment_task_efficient)

    logger.debug("\n--- 4スレッドで並列実行（GIL有効） ---")
    run_experiment(4, increment_task_efficient)
