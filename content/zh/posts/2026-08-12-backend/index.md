---
title: "Backend Basic"
date: 2026-08-12T12:00:00+08:00
author: "dengkoicat"
tags: ["Python", "Java", "Redis", "SQL"]
categories: ["技术博客"]
readingTime: 25
toc: true
ShowToc: true
TocOpen: false
draft: false
type: "posts"
---

## Asyncio

写后端时，`asyncio` 最容易被误解成“让 Python 变快”。更准确地说，它解决的是另一类问题：当程序大量时间花在等待网络、磁盘、数据库、缓存、第三方 API 时，不要让一个等待中的请求占住整条执行线。

同步程序的默认思路是：函数调用之后一直等，等它返回再继续下一步。异步程序的思路是：遇到可以等待的 I/O，就把执行权还给事件循环，让事件循环先去推进别的任务。CPU 并没有凭空变多，只是等待时间被更充分地利用了。

这也是为什么 `asyncio` 很适合 Web 服务、爬虫、消息队列消费者、长连接服务、批量调用外部 API 等场景；但如果任务主要是在本地做大规模计算，例如压缩图片、训练模型、解析巨型文件，`asyncio` 本身不会明显提升性能，应该考虑多进程、线程池、向量化计算或把任务丢给专门的 worker。

### Event Loop：一个线程里的调度器

`asyncio` 的核心是事件循环。它不是魔法，也不是自动并行执行所有代码的运行时。可以把它理解成一个负责调度协程的循环：

```markdown
                    asyncio
                       │
                       ▼
                  Event Loop
                       │
                调度很多 Task
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Task A          Task B          Task C
        │              │              │
        ▼              ▼              ▼
   Coroutine      Coroutine      Coroutine
        │              │              │
      await          await          await
        │              │              │
        └─────── 发生 I/O 等待 ───────┘
                       │
                       ▼
             把执行权还给 Event Loop
                       │
                       ▼
                 执行其他 Task
```

普通函数用 `def` 定义，调用后会立刻执行；协程函数用 `async def` 定义，调用后只会得到一个 coroutine object，并不会马上运行。真正让它运行的，是 `await`、`asyncio.run()`、`create_task()` 这类入口。

```python
import asyncio


async def fetch_user(user_id: int) -> dict:
    await asyncio.sleep(1)
    return {"id": user_id, "name": "Deng"}


async def main() -> None:
    user = await fetch_user(1)
    print(user)


asyncio.run(main())
```

这里的 `asyncio.sleep(1)` 可以理解成一个模拟 I/O 的等待点。执行到 `await` 时，当前协程暂停，事件循环可以去运行别的 task。等 1 秒计时结束后，这个协程再从暂停的位置继续执行。

因此，`await` 的含义不是“开一个新线程”，而是“我现在要等一个异步结果，等待期间可以把控制权交出去”。如果一个 `async def` 里面全是普通 CPU 计算，没有任何真正可让出的等待点，它依然会阻塞事件循环。

### Coroutine、Task 和 Future

学习 `asyncio` 时，最常见的三个词是 coroutine、task、future。它们不是同一个层级。

Coroutine 是“可暂停的函数调用”。调用 `async def` 函数得到的就是 coroutine，它描述了要做什么，但还没有被事件循环正式调度。

Task 是“已经交给事件循环调度的 coroutine”。一旦用 `asyncio.create_task()` 包起来，事件循环就可以在合适的时候推进它。

Future 是“未来某个时刻会完成的结果容器”。平时业务代码里不常手写 `Future`，但很多底层异步库会用它表示尚未完成的 I/O 结果。

```python
import asyncio


async def call_api(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name} done"


async def main() -> None:
    task = asyncio.create_task(call_api("profile", 1.0))

    print("task has been scheduled")

    result = await task
    print(result)


asyncio.run(main())
```

`create_task()` 之后，`call_api()` 已经被安排进事件循环。后面 `await task` 的意思是：当前协程需要这个任务的结果，如果还没完成，就先让出执行权。

一个容易踩坑的地方是：只创建 coroutine，不 `await`，也不创建 task，它不会真正执行。

```python
async def send_email() -> None:
    ...


async def main() -> None:
    send_email()  # 错误：这里只是创建 coroutine，没有执行
    await send_email()  # 正确：等待它执行完成
```

在真实项目里，这类问题经常表现为日志没打、请求没发、后台任务没跑，同时还可能出现 `coroutine was never awaited` 警告。

### 并发：`gather`、`create_task` 和 `TaskGroup`

`asyncio` 最直接的收益来自并发等待。比如一个后端接口需要同时查用户资料、订单列表和推荐结果，如果三件事彼此独立，就没有必要串行等待。

串行写法的问题很明显：

```python
import asyncio


async def get_profile() -> str:
    await asyncio.sleep(1)
    return "profile"


async def get_orders() -> str:
    await asyncio.sleep(1)
    return "orders"


async def get_recommendations() -> str:
    await asyncio.sleep(1)
    return "recommendations"


async def main() -> None:
    profile = await get_profile()
    orders = await get_orders()
    recommendations = await get_recommendations()

    print(profile, orders, recommendations)


asyncio.run(main())
```

这段代码总耗时接近 3 秒，因为每一步都等上一步完成。改成 `asyncio.gather()` 后，三个 I/O 等待可以重叠：

```python
async def main() -> None:
    profile, orders, recommendations = await asyncio.gather(
        get_profile(),
        get_orders(),
        get_recommendations(),
    )

    print(profile, orders, recommendations)
```

它的时间线更接近下面这样：

```markdown
profile          ██████████ 1s
orders           ██████████ 1s
recommendations  ██████████ 1s

total            ██████████ about 1s
```

`gather()` 适合“我有一组任务，并且需要等它们全部完成”的场景。它会按传入顺序返回结果，而不是按完成顺序返回。

如果想先启动一个任务，后面再决定什么时候等待，可以用 `create_task()`：

```python
async def main() -> None:
    profile_task = asyncio.create_task(get_profile())

    orders = await get_orders()
    profile = await profile_task

    print(profile, orders)
```

Python 3.11 之后，更推荐在结构化并发场景里使用 `TaskGroup`。它的好处是任务生命周期更清晰：离开 `TaskGroup` 作用域时，里面的任务要么全部正常结束，要么异常被集中抛出，未完成任务会被取消。

```python
import asyncio


async def main() -> None:
    async with asyncio.TaskGroup() as tg:
        profile_task = tg.create_task(get_profile())
        orders_task = tg.create_task(get_orders())
        rec_task = tg.create_task(get_recommendations())

    print(
        profile_task.result(),
        orders_task.result(),
        rec_task.result(),
    )


asyncio.run(main())
```

对于后端服务来说，`TaskGroup` 的语义更接近“这一组子任务属于当前请求”。请求结束时，子任务也应该被明确收束，而不是悄悄留在事件循环里继续跑。

不过，`gather()` 和 `TaskGroup` 更适合“一批任务同时开始，最后一起收口”的场景。如果任务是持续进入的，或者上游和下游速度不一致，就应该考虑 `asyncio.Queue`。Queue 的价值不是让代码更高级，而是把并发拆成生产者和消费者，并且通过 `maxsize` 做背压：下游处理不过来时，上游会在 `put()` 处等待，避免内存被无限撑大。

在后端里，Queue 常见于这几类场景：消息流消费、批量文件处理、爬虫 URL 调度、日志/埋点异步落库、RAG 文档处理流水线。它适合“任务之间有阶段关系”的流程，而不是简单的三次独立 API 调用。

以 RAG 管道为例，文档解析、Embedding 计算、向量库写入通常不是同一种并发模型。解析 PDF 可能是纯 Python CPU 密集，适合进程池；Embedding 如果调用的是本地 C/CUDA 库，可能适合线程池或专用服务；向量库写入大多是网络 I/O，直接 `await`。这时就不要等所有文档解析完再统一 embedding，而是让每个阶段通过 Queue 串起来，谁完成就把结果交给下一段。

```python
parse_queue = asyncio.Queue(maxsize=20)
embed_queue = asyncio.Queue(maxsize=20)

# parse_worker:
#   解析一个文档
#   await parse_queue.put(text)
#
# embed_worker:
#   text = await parse_queue.get()
#   计算 embedding
#   await embed_queue.put((text, embedding))
#
# store_worker:
#   text, embedding = await embed_queue.get()
#   await vector_db.upsert(text, embedding)
```

这个结构的关键是“流水线并发”：解析、Embedding、写入三个阶段可以同时推进，但每个阶段又能设置自己的并发上限。Queue 的 `maxsize` 负责背压，worker 数量负责吞吐，结束信号负责让消费者自然退出。相比一口气 `gather()` 几千个任务，这种写法更适合长批次任务和生产服务。

### 超时、取消和异常

异步代码里最不能省的是超时。一个外部 API、数据库查询或缓存连接如果没有超时，就可能把请求挂死。同步代码会卡住线程；异步代码虽然不会卡住整个线程，但会泄漏任务、堆积连接，并最终拖垮服务。

Python 3.11 以后可以用 `asyncio.timeout()` 给一段异步逻辑加超时：

```python
import asyncio


async def call_payment_api() -> str:
    await asyncio.sleep(10)
    return "ok"


async def main() -> None:
    try:
        async with asyncio.timeout(2):
            result = await call_payment_api()
            print(result)
    except TimeoutError:
        print("payment api timeout")


asyncio.run(main())
```

超时的本质通常是取消当前等待中的任务。取消不是普通失败，它会在协程内部抛出 `asyncio.CancelledError`。如果协程持有连接、锁、临时文件或其他资源，必须用 `try/finally` 做清理。

```python
async def consume_messages(consumer) -> None:
    await consumer.start()
    try:
        async for message in consumer:
            await handle_message(message)
    finally:
        await consumer.stop()
```

不要随手吞掉 `CancelledError`。如果确实要捕获它做清理，清理后通常应该继续抛出，让上层知道任务已经被取消。

```python
async def worker() -> None:
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        await flush_metrics()
        raise
```

`gather()` 的异常行为也要注意：默认情况下，只要其中一个 awaitable 抛异常，`gather()` 就会把第一个异常抛给调用方；其他已经提交的 awaitable 通常不会因为这个异常自动取消，但调用方也拿不到一组完整的正常结果。业务上如果需要“尽力返回”，可以显式使用 `return_exceptions=True`，但要认真处理返回列表里的异常对象。

```python
results = await asyncio.gather(
    get_profile(),
    get_orders(),
    get_recommendations(),
    return_exceptions=True,
)

for item in results:
    if isinstance(item, Exception):
        print("partial failure:", item)
    else:
        print("success:", item)
```

这种写法适合推荐位、画像补充、非核心信息聚合等场景；不适合支付、下单、扣库存这类强一致流程，因为局部失败不能被轻描淡写地吞掉。

### 后端项目里的实践边界

`asyncio` 在后端项目里通常不会单独出现，而是藏在框架和客户端库下面。比如 FastAPI、Starlette、aiohttp、asyncpg、aioredis、httpx 的异步客户端，本质上都是围绕事件循环组织 I/O。

一个典型接口可能长这样：

```python
from fastapi import FastAPI
import asyncio
import httpx

app = FastAPI()


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict:
    response = await client.get(url, timeout=2.0)
    response.raise_for_status()
    return response.json()


@app.get("/dashboard/{user_id}")
async def dashboard(user_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        profile, orders = await asyncio.gather(
            fetch_json(client, f"https://api.example.com/users/{user_id}"),
            fetch_json(client, f"https://api.example.com/orders/{user_id}"),
        )

    return {
        "profile": profile,
        "orders": orders,
    }
```

这里有几个实际约束。

第一，异步函数里不要调用会长时间阻塞的同步 I/O。例如 `requests.get()`、普通数据库同步客户端、大文件同步读写，都可能把事件循环卡住。既然入口已经是 `async def`，依赖也应该尽量换成异步版本。

第二，连接池要复用。上面的例子为了简洁把 `AsyncClient` 写在请求内部，真实服务里更常见的是在应用启动时创建客户端，在关闭时统一释放。数据库连接池、HTTP 连接池、Redis 连接池也是同理。

第三，控制并发量。`asyncio.gather()` 可以一次性发起很多任务，但外部系统不一定扛得住。批量请求第三方服务时，应该用 `Semaphore` 做限流。

```python
import asyncio
import httpx

limit = asyncio.Semaphore(10)


async def fetch_with_limit(client: httpx.AsyncClient, url: str) -> dict:
    async with limit:
        response = await client.get(url, timeout=2.0)
        response.raise_for_status()
        return response.json()
```

第四，区分 `await`、线程池和进程池。判断标准不是“这个函数写在 async 里面”，而是它等待的到底是什么。

如果调用对象本身就是异步 I/O，例如异步 HTTP 客户端、异步数据库驱动、异步 Redis 客户端，就直接 `await`。这类等待不会占住事件循环，正是 `asyncio` 最擅长的地方。

```python
response = await client.get(url, timeout=2.0)
rows = await db.fetch(query)
await redis.set(key, value)
```

如果调用对象是阻塞式 I/O，或者是没有 async 版本的老 SDK，例如 `requests`、同步数据库客户端、同步文件处理、某些云厂商 SDK，就不要直接在事件循环里调用。轻量阻塞任务可以先放进线程池，避免卡住整个 async 服务。

```python
import asyncio


def parse_large_file(path: str) -> dict:
    ...


async def handle_upload(path: str) -> dict:
    return await asyncio.to_thread(parse_large_file, path)
```

`asyncio.to_thread()` 是最省事的线程池入口，适合“偶尔包一层同步函数”。如果需要控制线程数量，或者同一类任务会大量出现，可以显式使用 `ThreadPoolExecutor`。

```python
from concurrent.futures import ThreadPoolExecutor

thread_pool = ThreadPoolExecutor(max_workers=8)

result = await loop.run_in_executor(thread_pool, blocking_io_call, arg)
```

如果任务是纯 Python CPU 密集，例如复杂文本解析、压缩解压、图片处理、规则引擎、大量 JSON 反序列化，线程池通常帮不大，因为 GIL 会限制同一进程内 Python 字节码的并行执行。这类任务更适合进程池，或者直接交给独立 worker 服务。

```python
from concurrent.futures import ProcessPoolExecutor

process_pool = ProcessPoolExecutor(max_workers=4)

text = await loop.run_in_executor(process_pool, parse_pdf, path)
```

可以用一个简单规则记住：

- 异步 I/O：直接 `await`。
- 阻塞 I/O：用线程池，或者换成 async 客户端。
- 纯 Python CPU 密集：用进程池、任务队列或独立计算服务。
- C 扩展/GPU 计算：看库是否释放 GIL；如果释放，线程池可能有效，否则仍然考虑进程池或服务化。

RAG 管道就是一个典型混合场景：文档解析可能进程池，Embedding 可能线程池或模型服务，向量库写入直接 `await`。`asyncio` 在这里负责统一调度，而不是亲自完成所有计算。

最后，可以用一个简单判断来决定要不要引入 `asyncio`：如果瓶颈主要来自等待 I/O，并且一次请求或一个任务里有多个可重叠的等待点，异步会让吞吐明显改善；如果瓶颈主要来自 CPU，异步只会让代码结构更复杂。`asyncio` 的价值不是“所有地方都 async”，而是在等待很多、连接很多、任务很多的时候，让一个线程更有秩序地处理这些等待。
